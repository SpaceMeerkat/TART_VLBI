import numpy as np
import pandas as pd
from scipy.fft import fft, ifft
from multiprocessing import Pool
from tqdm import tqdm
from IPython.display import display, HTML

def _acquire_worker(args):
    """Top-level worker function for multiprocessing Pool (must be picklable)."""
    tart_name, sat_id, voltages_bb_lpf, prn_upsampled, Fs = args
    acq = acquire_GPS(Fs)
    result = acq.acquire_satellite(voltages_bb_lpf, prn_upsampled, Fs)
    return tart_name, sat_id, result

def _acquire_worker_chunked(args):
    """Top-level worker for chunked multiprocessing Pool (must be picklable)."""
    tart_name, sat_id, tart, last_idx, N_chunks, prn_upsampled, Fs = args
    print(f"[{tart_name}] Starting PRN {sat_id}", flush=True)
    acq = acquire_GPS(Fs)
    result = acq.acquire_satellite_chunked(tart, last_idx, N_chunks, prn_upsampled, Fs,
                                           tart_name=tart_name, sat_id=sat_id)
    print(f"[{tart_name}] Done PRN {sat_id} — detected: {result['detected']}, doppler: {result['doppler']:.1f} Hz", flush=True)
    return tart_name, sat_id, result

class acquire_GPS:

    # Speed of light in m/s
    C = 299792458.0

    # GPS L1 C/A code parameters
    CHIP_RATE = 1.023e6  # 1.023 MHz chip rate
    CODE_LENGTH = 1023   # chips per PRN period
    PRN_PERIOD = 1e-3    # 1 ms PRN period

    # G2 tap positions for GPS satellites 1-32 (Gold code generation)
    G2_TAPS = [
        (2, 6), (3, 7), (4, 8), (5, 9), (1, 9), (2, 10), (1, 8), (2, 9),
        (3, 10), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10),
        (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9), (1, 3), (4, 6),
        (5, 7), (6, 8), (7, 9), (8, 10), (1, 6), (2, 7), (3, 8), (4, 9)
    ]

    def __init__(self, Fs=0):
        """
        Initialize SignalProcessor with sampling frequency.

        Args:
            Fs: Sampling frequency in Hz (e.g., 16.368e6 for TART)
        """
        self.Fs = Fs
        self.samples_per_chip = Fs / self.CHIP_RATE
        self.prn_period_samples = int(Fs * self.PRN_PERIOD)

    # =========================================================================
    # PRN Code Generation
    # =========================================================================

    def generate_ca_prn(self, prn_num):
        """
        Generate the 1023-chip GPS C/A PRN sequence for a given satellite.

        Uses the Gold code structure with G1 and G2 linear feedback shift registers.

        Args:
            prn_num: Satellite PRN number (1-32)

        Returns:
            np.ndarray: Bipolar (+1/-1) PRN sequence of length 1023
        """
        if not (1 <= prn_num <= 32):
            raise ValueError(f"PRN must be 1-32, got {prn_num}")

        # Initialize shift registers with all ones
        g1 = np.ones(10, dtype=int)
        g2 = np.ones(10, dtype=int)
        ca = np.zeros(self.CODE_LENGTH, dtype=int)

        # Get tap positions for this PRN
        tap1, tap2 = self.G2_TAPS[prn_num - 1]

        for i in range(self.CODE_LENGTH):
            # Output is XOR of G1 output and two G2 taps
            ca[i] = g1[-1] ^ (g2[tap1-1] ^ g2[tap2-1])

            # Shift G1: feedback from positions 3 and 10 (indices 2 and 9)
            newbit_g1 = g1[2] ^ g1[9]
            g1[1:] = g1[:-1]
            g1[0] = newbit_g1

            # Shift G2: feedback from positions 2,3,6,8,9,10
            newbit_g2 = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]
            g2[1:] = g2[:-1]
            g2[0] = newbit_g2

        # Convert to +/- 1 voltages: 0 -> +1, 1 -> -1
        return 1 - 2 * ca

    def generate_all_prn_codes(self):
        samples_per_ms = int(self.Fs / 1000)            
        # Pre-compute all PRN codes once (OPTIMIZATION)
        print("Pre-computing PRN codes for all satellites...")
        prn_codes = {}
        for prn_num in range(1, 33):
            chips = self.generate_ca_prn(prn_num)
            # upsampled = np.repeat(chips, self.samples_per_chip)[:samples_per_ms]
            t_chip = np.arange(1023) / 1.023e6
            t_samp = np.arange(samples_per_ms) / 16368000
            upsampled = np.interp(t_samp, t_chip, chips, period=1e-3)
            
            prn_codes[prn_num] = upsampled.astype(np.complex64)
        print(f"Done.")
        return prn_codes

    def acquire_satellite_chunked(self, data, last_idx, N_chunks, prn_upsampled, Fs,
                          tart_name='', sat_id='',
                          doppler_range=np.arange(-5000, 5001, 250)):

        samples_per_ms = int(Fs / 1000)
        PRN_fft = fft(prn_upsampled[:samples_per_ms])
        t = np.arange(samples_per_ms, dtype=np.float32) / Fs

        antenna_results = {}
        for ant_idx in data['antenna_selection']:
            print(f"  [{tart_name}] PRN {sat_id} — antenna {ant_idx}", flush=True)
            chunk_SNRs = np.ones(int(N_chunks)) * -1
            chunk_dopplers = np.zeros(int(N_chunks))
            correlations = np.zeros((int(N_chunks), samples_per_ms))

            for chunk_idx in range(int(N_chunks)):
                chunk = np.split(data['data'][ant_idx][:last_idx], N_chunks)[chunk_idx]
                signal = chunk.astype(np.complex64)
                for f_doppler in doppler_range:
                    segment = signal * np.exp(-1j * 2 * np.pi * f_doppler * t)
                    corr = ifft(fft(segment) * np.conj(PRN_fft))
                    power = np.abs(corr)**2
                    SNR = np.max(power) / np.std(power)
                    if SNR > chunk_SNRs[chunk_idx]:
                        chunk_SNRs[chunk_idx] = SNR
                        chunk_dopplers[chunk_idx] = f_doppler
                        correlations[chunk_idx] = power

            antenna_results[ant_idx] = {
                'chunk_SNRs': chunk_SNRs,
                'chunk_dopplers': chunk_dopplers,
                'correlations': correlations,
            }

        best_antenna = max(antenna_results, key=lambda a: np.median(antenna_results[a]['chunk_SNRs']))
        best_dopplers = antenna_results[best_antenna]['chunk_dopplers']
        detected = bool(np.std(best_dopplers) < 750) if len(data['antenna_selection']) > 1 else True

        return {'antenna_results': antenna_results,
                'detected': detected, 
                'doppler': np.median(best_dopplers),
                'snrs': np.median(antenna_results[best_antenna]['chunk_SNRs']), 
                'best_antenna': best_antenna}

    def acquire_satellite(self, voltages_bb_lpf, prn_upsampled, Fs,
                          doppler_range=np.arange(-5000, 5001, 50),
                          T_coh_ms=1, N_noncoh=20):

        samples_per_ms = int(Fs / 1000)
        N_coh = T_coh_ms * samples_per_ms
        N_total = N_coh * N_noncoh
        PRN_fft = fft(prn_upsampled)  # Pre-compute FFT (OPTIMIZATION)
        antenna_snrs = np.zeros(24)
        raw_detected_doppler = np.zeros(24)
        detected = False
        detected_doppler = 0
        best_global_snr = 0
        best_antenna = 0

        t = np.arange(N_coh, dtype=np.float32) / Fs
        for ant_idx in range(24):
            signal = voltages_bb_lpf[ant_idx][:N_total].astype(np.complex64)
            snr_intermediate = np.zeros(len(doppler_range))
            for idx, f_doppler in enumerate(doppler_range):
                noncoh_power = np.zeros(samples_per_ms, dtype=np.float32)
                for k in range(N_noncoh):
                    segment = signal[k*N_coh:(k+1)*N_coh]
                    segment = segment * np.exp(-1j * 2 * np.pi * f_doppler * t)
                    corr = ifft(fft(segment) * np.conj(PRN_fft))
                    noncoh_power += np.abs(corr)**2
                peak = np.max(noncoh_power)
                noise = np.median(noncoh_power)
                snr = peak / noise if noise > 0 else 0
                snr_intermediate[idx] = snr
                if snr >= best_global_snr:
                    best_global_snr = snr
                    detected_doppler = f_doppler
                    best_antenna = ant_idx
            max_snr_idx = np.argmax(snr_intermediate)
            raw_detected_doppler[ant_idx] = float(doppler_range[max_snr_idx])
    
        for ant_idx in range(24):
            signal = voltages_bb_lpf[ant_idx][:N_total].astype(np.complex64)
            noncoh_power = np.zeros(samples_per_ms, dtype=np.float32)
            for k in range(N_noncoh):
                segment = signal[k*N_coh:(k+1)*N_coh]
                segment = segment * np.exp(-1j * 2 * np.pi * detected_doppler * t)
                corr = ifft(fft(segment) * np.conj(PRN_fft))
                # corr = ifft(fft(segment, 2*N) * np.conj(fft(prn, 2*N)))
                noncoh_power += np.abs(corr)**2
            peak = np.max(noncoh_power)
            noise = np.median(noncoh_power)
            antenna_snrs[ant_idx] = peak / noise if noise > 0 else 0
        if int(sum(antenna_snrs > 5)) >= 2:
            detected = True
        return {'detected': detected, 'doppler': detected_doppler, 'raw_detected_doppler': raw_detected_doppler, 
                'snrs': antenna_snrs, 'best_antenna': best_antenna}

    def acquire_satellites_pool(self, TART_DATA, visible_satellites, prn_codes, n_workers=None):
        """Pool-friendly parallel version of acquire_satellites.

        Args:
            TART_DATA: dict of {tart_name: {'data': voltages, 'Fs': sample_rate}}
            visible_satellites: list of satellite IDs (e.g. ['G01', 'G10', ...])
            prn_codes: dict of {prn_num: upsampled_code}
            n_workers: number of Pool workers (defaults to cpu count)

        Returns:
            dict: results[tart_name][sat_id][antenna] -> chunked acquisition result
        """
        chunk_duration = 1e-3  # 1 ms per chunk

        # Pre-compute chunk params once per TART to avoid repeating for every (sat, antenna) job
        tart_chunk_params = {}
        for tart_name, tart in TART_DATA.items():
            Fs = tart['Fs']
            chunk_samples = chunk_duration * Fs
            N_chunks = len(tart['data'][0]) // chunk_samples
            last_idx = int(N_chunks * chunk_samples)
            tart_chunk_params[tart_name] = (last_idx, int(N_chunks))

        work_items = []
        for sat_id in visible_satellites:
            prn_num = int(str(sat_id)[1:])
            prn = prn_codes[prn_num]
            for tart_name, tart in TART_DATA.items():
                last_idx, N_chunks = tart_chunk_params[tart_name]
                work_items.append((tart_name, sat_id, tart, last_idx, N_chunks, prn, tart['Fs']))

        results = {tart_name: {} for tart_name in TART_DATA}
        print(f"Running parallel chunked acquisition for {len(visible_satellites)} satellites ({len(work_items)} jobs, {n_workers or 'auto'} workers)...")
        print("=" * 60)
        with Pool(processes=n_workers) as pool:
            for tart_name, sat_id, result in tqdm(
                pool.imap_unordered(_acquire_worker_chunked, work_items),
                total=len(work_items),
                desc="Acquiring satellites",
            ):
                results[tart_name][sat_id] = result
        print("=" * 60)
        print("Acquisition complete!")
        return results

    def acquire_satellites(self, TART_DATA, visible_satellites, prn_codes):
        results = {}  # results[TART][sat_id] -> acquisition result
        print(f"Running acquisition for {len(visible_satellites)} satellites..."); print("=" * 60);
        for tart_name in TART_DATA:
            results[tart_name] = {}
        for sat_id in tqdm(visible_satellites, desc="Acquiring satellites"):
            prn_num = int(str(sat_id)[1:]) 
            prn = prn_codes[prn_num]
            for tart_name, tart in TART_DATA.items():
                results[tart_name][sat_id] = self.acquire_satellite(tart['data'], prn, tart['Fs'])
        print("=" * 60); print("Acquisition complete!")
        return results

    def filter_acquired_satellites(self, acquisition_results, viable_satellites, TARTS):
        """ 
        Remove satellites from the viable satellites list which were not sufficiently detected in doppler
        """
        filtered_satellites = np.copy(viable_satellites)
        for satellite in viable_satellites:
            for TART in TARTS.keys():
                if acquisition_results[TART][satellite]['detected'] != True:
                    filtered_satellites = filtered_satellites[~(filtered_satellites == satellite)]
                    continue
        return filtered_satellites

    def snr_tables(self, acquisition_results, satellite_names, tart_names, SNR_limit=5):
        """Build and display one SNR table per satellite.

        Each table has one row per TART site. Columns are the union of all antenna
        indices present across TARTs; antennas not used by a given TART show as NaN.

        Args:
            acquisition_results: dict of {tart_name: {sat_id: {'antenna_results': {ant_idx: {...}}, ...}}}
            satellite_names: list/array of satellite IDs (e.g. ['G23', 'G28', 'G31'])
            tart_names: list of TART site names (e.g. ['rhodes', 'namibia'])
            SNR_limit: threshold below which cells are highlighted red (default 5)

        Returns:
            list of pandas Styler objects (one per satellite)
        """
        def _highlight_low_snr(val):
            if pd.isna(val):
                return "background-color: #d0d0d0"  # grey for antennas not used by this TART
            if val < SNR_limit:
                return "background-color: #e8a0a0"  # desaturated red
            return ""

        styled_tables = []
        for sat_id in satellite_names:
            sat_id = str(sat_id)

            # Collect median chunk SNR per antenna for each TART, and find union of all antenna indices
            tart_antenna_snrs = {}
            all_antennas = set()
            for tart_name in tart_names:
                ant_results = acquisition_results[tart_name][sat_id]['antenna_results']
                tart_antenna_snrs[tart_name] = {
                    ant_idx: np.median(ant_data['chunk_SNRs'])
                    for ant_idx, ant_data in ant_results.items()
                }
                all_antennas.update(ant_results.keys())

            sorted_antennas = sorted(all_antennas)
            antenna_cols = [f"Ant {i}" for i in sorted_antennas]

            rows = []
            for tart_name in tart_names:
                row = [tart_antenna_snrs[tart_name].get(a, np.nan) for a in sorted_antennas]
                rows.append(row)

            df = pd.DataFrame(rows, index=list(tart_names), columns=antenna_cols)
            styler = (
                df.style
                .format("{:.2f}", na_rep="—")
                .map(_highlight_low_snr)
                .set_caption(f"Satellite {sat_id}  (SNR limit = {SNR_limit})")
            )
            styled_tables.append(styler)
            display(styler)

        return styled_tables