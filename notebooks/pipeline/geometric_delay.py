import numpy as np

class geometry:

    def __init__(self):
        self.C = 299792458.0
        return

    def compute_inter_tart_delays(
        self,
        visible_satellites: dict,
        tart_positions: dict,
        tart_ref: str,
        ref_antennas: dict,   # now indexed by sat_id
    ):
    
        delays = {}
    
        for sat_id, s in visible_satellites.items():
            sat_delays = {}
    
            # Reference antenna for THIS satellite
            ref_ant_idx = ref_antennas[sat_id][tart_ref]
            A_ref = tart_positions[tart_ref][ref_ant_idx]
    
            # Distance to reference TART reference antenna
            d_ref = np.linalg.norm(A_ref - s)
    
            for tart_name, antennas in tart_positions.items():
                if tart_name == tart_ref:
                    continue
    
                ant_idx = ref_antennas[sat_id][tart_name]
                B_ref = antennas[ant_idx]
    
                d_tart = np.linalg.norm(B_ref - s)
    
                delay = float((d_ref - d_tart) / self.C)
                sat_delays[tart_name] = delay
    
            delays[sat_id] = sat_delays
    
        return delays