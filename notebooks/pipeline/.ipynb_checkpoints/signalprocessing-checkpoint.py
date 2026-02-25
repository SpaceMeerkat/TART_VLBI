import numpy as np
from scipy.signal import resample_poly, correlate
from scipy.ndimage import shift as ndshift

class signal_processing_handler:

    def __init__(self):
        self.CARRIER_FREQ = 1575.42e6  # L1
        self.CODE_RATE = 1.023e6 # GPS L1 C/A
        return 

    def correlate_with_prn(self, voltage, prn_filter):
        """
        Correlate voltage signal with upsampled PRN code using FFT.

        Args:
            voltage: Preprocessed voltage signal (1D array)
            prn_upsampled: Upsampled PRN code (1D array)

        Returns:
            np.ndarray: Complex correlation output (same length as voltage)
        """
        n = len(voltage) + len(prn_filter) - 1
        # Use next power of 2 for efficient FFT
        nfft = 1 << int(np.ceil(np.log2(n)))

        # FFT-based correlation
        V = np.fft.fft(voltage, n=nfft)
        P = np.fft.fft(prn_filter, n=nfft)

        corr = np.fft.ifft(V * np.conj(P))

        return corr[:len(voltage)]

    def get_correlations(self, TART, TART_DATA, Fs_baseband, acquisition_results, ref_antennas, satellite, prn_filters, interpolation_factor=1): 
        antenna_data = TART_DATA[TART]['data'][ref_antennas[satellite][TART]]
        doppler_det = acquisition_results[TART][satellite]['doppler']
        code_doppler = doppler_det * (self.CODE_RATE / self.CARRIER_FREQ)
        code_rate_eff = self.CODE_RATE + code_doppler
        n_samples_original = len(antenna_data)
        n_samples_interp = n_samples_original * interpolation_factor
        # Create Doppler compensation carrier at interpolated rate
        Fs_interp = Fs_baseband * interpolation_factor
        t_interp = np.arange(n_samples_interp) / Fs_interp
        doppler_carrier = np.exp(-1j * 2 * np.pi * doppler_det * t_interp).astype(np.complex64)
        # Interpolate real baseband signal
        voltage_interp = resample_poly(antenna_data, interpolation_factor, 1).astype(np.complex64)
        # Ensure correct length
        if len(voltage_interp) > n_samples_interp:
            voltage_interp = voltage_interp[:n_samples_interp]
        elif len(voltage_interp) < n_samples_interp:
            voltage_interp = np.pad(voltage_interp, (0, n_samples_interp - len(voltage_interp)))
        # Apply Doppler compensation (makes it complex)
        voltage_compensated = voltage_interp * doppler_carrier
        # Correlate with PRN code
        prn_nominal = prn_filters[int(satellite[1:])]
        corr = self.correlate_with_prn(voltage_compensated, prn_nominal)
        # Store magnitude
        antenna_correlation = np.abs(corr).astype(np.float32)
        del corr, voltage_compensated, voltage_interp
        return antenna_correlation

    def delay_to_samples(self, delay_seconds, Fs, interpolation_factor=1):
        """
        Convert delay in seconds to samples at (optionally interpolated) rate.

        Args:
            Fs: Sampling frequency in Hz
            delay_seconds: Delay in seconds
            interpolation_factor: Upsampling factor applied to data

        Returns:
            float: Delay in samples
        """
        effective_fs = Fs * interpolation_factor
        return delay_seconds * effective_fs

    def apply_delay_shift(self, signal, delay_samples):
        """
        Shift signal by a fractional number of samples.

        Positive delay_samples shifts signal to the right (later in time).
        Uses spline interpolation for sub-sample accuracy.

        Args:
            signal: 1D input signal
            delay_samples: Shift amount (can be fractional)

        Returns:
            np.ndarray: Shifted signal (same length as input)
        """
        if delay_samples == 0:
            print("No shift detected, returing original signal")
            return signal.copy()
        return ndshift(signal, delay_samples, mode='constant', cval=0)

    def apply_delay(self, signal, delay_seconds, Fs, interpolation_factor=1):
        shift_samples = self.delay_to_samples(delay_seconds, Fs, interpolation_factor)
        shited_signal = self.apply_delay_shift(signal, shift_samples)
        return shited_signal