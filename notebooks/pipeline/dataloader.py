import h5py
import numpy as np
import json
from datetime import datetime
import os
from scipy.signal import firwin, lfilter

class raw_data_loader:
    """ 
    Class for handling loading the raw TART voltage data and preprocessing.
    """

    def __init__(self, basepath):
        self.basepath = basepath
        self.cutoff_hz = 2.0e6
        self.num_taps = 129
        return

    def parse_hdf5_metadata(self, filepath):
        """
        Opens an HDF5 file and extracts the critical data and metadata.
        Returns None if the file cannot be opened or parsed.
        """
        try:
            if not os.path.exists(filepath):
                 print(f"ERROR: File not found at: {filepath}")
                 return None
    
            with h5py.File(filepath, 'r') as f:

                config_bytes = f['config'][()][0]
                config_string = config_bytes.decode('utf-8')
                config_dict = json.loads(config_string)
                Fs = float(config_dict['sampling_frequency'])
                
                timestamp_bytes = f['timestamp'][()][0]
                timestamp_string = timestamp_bytes.decode('utf-8')
                T0_datetime = datetime.fromisoformat(timestamp_string)
                T0_seconds = T0_datetime.timestamp()
                
                packed_data = f['data'][:] 
                voltage_data = np.unpackbits(packed_data, axis=-1)
                
                return {
                    'T0_utc': T0_datetime,
                    'T0_seconds': T0_seconds,
                    'Fs': Fs,
                    'data': voltage_data,
                    'filepath': filepath
                }
        except Exception as e:
            print(f"ERROR: Failed to process HDF5 content in {filepath}: {e}")
            return None

    def trim_byte_padding(self, data):
        """
        Trims the end of the raw voltages where they stop recording
        """
        assert data.shape[0] == 24, "data.shape[0] is not 24 (i.e. the expected number of antennas)"
        padding_values = data[:,-1]
        diff = (data != padding_values[:, None])
        rev_diff = diff[:, ::-1]
        idx_from_end = np.argmax(rev_diff, axis=1)
        trim_index = idx_from_end.max()
        return data[:,:-trim_index]

    def mixdown_to_baseband(self, data):
        """
        Take the IF voltages and mix them down to baseband using the Fs/4 = baseband trick
        """
        n_samples = data['data'].shape[1]
        # Exact Fs/4 complex mixer:
        # exp(-j*pi*n/2) = [1, -j, -1, +j, ...]
        mix_i = np.tile([1, 0, -1, 0], n_samples // 4 + 1)[:n_samples].astype(np.float32)
        mix_q = np.tile([0, -1, 0, 1], n_samples // 4 + 1)[:n_samples].astype(np.float32)
        # Store complex baseband voltages
        voltages_baseband = np.zeros((24, n_samples), dtype=np.complex64)
        for ant_idx in range(24):
            # Convert 0/1 to bipolar ±1
            voltage_bipolar = 2.0 * data['data'][ant_idx].astype(np.float32) - 1.0
            I = voltage_bipolar * mix_i
            Q = voltage_bipolar * mix_q
            voltages_baseband[ant_idx] = I + 1j * Q
        return voltages_baseband

    def apply_low_pass_filter(self, data, Fs_baseband, cutoff_hz, num_taps):  
        """
        Clean up the baseband data by applying a low pass filter at +/- cutoff_hz
        """
        lpf = firwin(num_taps,cutoff=cutoff_hz,fs=Fs_baseband,window="hamming")
        # Apply per antenna
        voltages_bb_lpf = np.zeros_like(data)
        for ant_idx in range(24):
            voltages_bb_lpf[ant_idx] = lfilter(
                lpf, 1.0, data[ant_idx]
            )
        return voltages_bb_lpf

    def construct_time_axis(self, metadata):
        """
        Make time axes at the same interval as the voltage data
        """
        assert type(metadata) == dict, "metadata must be given as a dict using dataloader.parse_hdf5_metadata()"
        T_start_sec = metadata['T0_seconds']
        N = metadata['data'].shape[1]
        dt = 1.0 / metadata['Fs']
        return T_start_sec + np.arange(N) * dt

    def get_overlapping_time_window(self, time_arrays):
        """
        Gets the overlapping time limits for a list of time axes of arbitrary length
        """
        lower = max(arr.min() for arr in time_arrays)
        upper = min(arr.max() for arr in time_arrays)
        return [float(lower), float(upper)]

    def load_preprocessed_data(self, path):
        """
        Load and preprocess raw voltage datasets
        """
        data = self.parse_hdf5_metadata(f"{self.basepath}/{path}")
        print(f"Loading: {path}...")
        Fs = data['Fs']
        data['data'] = self.trim_byte_padding(data['data'])
        data['data'] = self.mixdown_to_baseband(data)
        data['data'] = self.apply_low_pass_filter(data['data'], Fs, self.cutoff_hz, self.num_taps)
        timeAxis = self.construct_time_axis(data)
        print(f"Completed.")
        return data, timeAxis, Fs