import astropy.time
import sp3
import astropy.units as u
import numpy as np
import traceback
import logging
logging.getLogger().setLevel(logging.ERROR)

class GPS_handler:
    """
    Class primarily for getting the positions of the 32 GPS satellites at some observational time t_obs.
    """

    def __init__(self):
        self.c = 299792458.0 # Speed of light
        self.omega_e = 7.2921151467e-5  # Earth rotation speed rad/s
        self.tau = 0.07 # Initial guess at light time for iterative position search
        return

    def GPS_positions(self, sat_ids, t_obs, rec_ecef):
        """
        Given the satellite IDs, the observation time, and a reference Earth location: fetches the GPS positions.
        Performs an iterative calculation of the exact positions given some initial guess.
        Applies Earth rotation effects.
        """
        
        record_positions = np.zeros((len(sat_ids), 3)) * np.nan
        
        for j, sat_id in enumerate(sat_ids):
            print(f"Searching for position of GPS satellite: {sat_id}...")
            try:
                # We use the first sample time for the calculation
                t_rec = t_obs
                tau = self.tau  # initial 70ms guess
                
                sat_pos_corr = np.array([0.0, 0.0, 0.0])
                
                # Iterative Light-Time + Sagnac Correction
                for _ in range(3):
                    t_trans = t_rec - astropy.time.TimeDelta(tau, format='sec')
                    
                    # Get SP3 position (Returns ITRS by default)
                    # Note: sp3 library coordinates are usually in METERS. 
                    pos_obj = sp3.itrs(
                        id=sp3.Sp3Id(sat_id), 
                        obstime=t_trans, 
                        download_directory="sp3_cache"
                    )
                    
                    # Convert CartesianRepresentation to numpy array
                    sat_pos = np.array([pos_obj.x.value, pos_obj.y.value, pos_obj.z.value])
                    
                    # Sagnac Effect: Rotate sat position to frame at t_rec
                    theta = self.omega_e * tau
                    cos_t, sin_t = np.cos(theta), np.sin(theta)
                    
                    x_corr = sat_pos[0] * cos_t + sat_pos[1] * sin_t
                    y_corr = -sat_pos[0] * sin_t + sat_pos[1] * cos_t
                    z_corr = sat_pos[2]
                    
                    sat_pos_corr = np.array([float(x_corr), float(y_corr), float(z_corr)])
                    
                    # Update travel time for next iteration
                    dist = np.linalg.norm(sat_pos_corr - rec_ecef)
                    tau = dist / self.c
                
                record_positions[j] = sat_pos_corr
                print(f"{sat_id:<5} | {sat_pos_corr[0]:>14.3f} | {sat_pos_corr[1]:>14.3f} | {sat_pos_corr[2]:>14.3f}")
                # print(f"Successfully retrieved {sat_id}")

            except Exception as e:
                # If 404 persists, the SP3 file for this date might not be released yet
                # or the CDDIS mirror is down.
                traceback.print_exc()
                print(e)
                print(f"Failed for satellite {sat_id}.")
                continue
                
        return record_positions

    def satellite_id_list(self):
        """ 
        Creates a list of satellite IDs
        """
        sat_ids = [f"G{i}" for i in range(1,33)]
        for i, id in enumerate(sat_ids):
            if len(id) == 2:
                sat_ids[i] = f"G0{id[-1]}"
        return sat_ids

    def calculate_elevation(self, sat_pos, site_pos):
        """
        Elevation angle (degrees) from a site to a satellite, both in ECEF
        """
        range_vec = np.array(sat_pos) - np.array(site_pos)
        up_vec = np.array(site_pos) / np.linalg.norm(site_pos)
        elevation_rad = np.arcsin(np.dot(range_vec, up_vec) / np.linalg.norm(range_vec))
        return np.degrees(elevation_rad)

    def calculate_viable_satellites(self, TART_positions, GPS_positions_dict, ELEVATION_LIMIT=15): 
        """
        Calculates whether satellites are above the horizon elevation limit (viable) and returns a dict
        """
        viable_satellites = {}
        for TART in TART_positions.keys():
            viable_satellites[TART] = []
        
        for sat_id, sat_position in GPS_positions_dict.items():
            for TART in TART_positions.keys():
                elevation = self.calculate_elevation(sat_position, TART_positions[TART]['station_ecef'])
                if elevation >= ELEVATION_LIMIT:
                    viable_satellites[TART].append(sat_id)
        return viable_satellites

    def get_union_satellites(self, viable_satellites, TARTS):
        viable_satellites_union = list(set.intersection(*(set(viable_satellites[n]) for n,_ in TARTS.items())))
        viable_satellites_union.sort()
        return viable_satellites_union

    def create_GPS_position_dict(self, sat_ids, GPS_positions):
        """
        Creates a dictionary of sat_id:sat_position key:value pairs.
        """
        GPS_positions_dict = {}
        for i in range(len(sat_ids)):
            GPS_positions_dict[sat_ids[i]] = GPS_positions[i].astype(float)
        return GPS_positions_dict

    def forward(self, t_obs, TART_ecef):
        """ 
        For a given observation time (t_obs), and reference Earth position (TART_ecef), get the GPS satellite positions
        """
        t_obs = astropy.time.Time(t_obs, format='unix')
        sat_ids = self.satellite_id_list()
        GPS_positions = self.GPS_positions(sat_ids, t_obs, TART_ecef)
        GPS_positions_dict = self.create_GPS_position_dict(sat_ids, GPS_positions)
        return GPS_positions_dict



















    
        