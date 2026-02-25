import numpy as np
import json
from datetime import datetime
import os
import requests
from astropy.coordinates import EarthLocation
import astropy.units as u

class TART_positions:

    def __init__(self):
        self.C = 299_792_458.0
        return
    
    def get_antenna_positions(self, telescope):
        url = f"https://api.elec.ac.nz/tart/{telescope}/api/v1/imaging/antenna_positions"
        headers = {
            'accept': 'application/json'
        }
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status() 
            data = response.json() 
            print(f"Antenna positions API request successful for {telescope}.")
            return np.array(data)
        except requests.exceptions.HTTPError as errh: print(f"HTTP Error: {errh}")
        except requests.exceptions.ConnectionError as errc: print(f"Connection Error: {errc}")
        except requests.exceptions.Timeout as errt: print(f"Timeout Error: {errt}")
        except requests.exceptions.RequestException as err: print(f"An unexpected error occurred: {err}")
        return

    def get_TART_details(self, telescope):
        url = f"https://api.elec.ac.nz/tart/{telescope}/api/v1/info"
        headers = {
            'accept': 'application/json'
        }
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status() 
            data = response.json() 
            print(f"TART position API request successful for {telescope}.")
            return data
        except requests.exceptions.HTTPError as errh: print(f"HTTP Error: {errh}")
        except requests.exceptions.ConnectionError as errc: print(f"Connection Error: {errc}")
        except requests.exceptions.Timeout as errt: print(f"Timeout Error: {errt}")
        except requests.exceptions.RequestException as err: print(f"An unexpected error occurred: {err}")
        return None

    def get_TART_positions_numpy(self, TART_metadata):
        TART_pos = np.array([
            TART_metadata['info']['location']['lon'],
            TART_metadata['info']['location']['lat'],
            TART_metadata['info']['location']['alt']
        ])
        return TART_pos

    def lat_long_alt_to_ECEF(self, latlonalt):
        lon, lat, alt = latlonalt[0], latlonalt[1], latlonalt[2]
        location = EarthLocation(
            lon=lon * u.deg,
            lat=lat * u.deg,
            height=alt * u.m
        )
        pos_ECEF = location.to_value(u.m)
        pos_ECEF = np.array([
            pos_ECEF['x'],
            pos_ECEF['y'],
            pos_ECEF['z']
        ])
        return pos_ECEF

    def get_TART_positions_ECEF(self, TART_name):
        TART_antenna_positions = self.get_antenna_positions(TART_name)
        TART_metadata = self.get_TART_details(TART_name)
        TART_pos_numpy = self.get_TART_positions_numpy(TART_metadata)
        TART_ECEF = self.lat_long_alt_to_ECEF(TART_pos_numpy)
        TART_antennas_ECEF = self.antennas_offset_to_ECEF(TART_ECEF, TART_pos_numpy, TART_antenna_positions)
        return TART_ECEF, TART_antennas_ECEF
        
    def enu_to_ecef_matrix(self, lat_deg, lon_deg):
        lat = np.deg2rad(lat_deg)
        lon = np.deg2rad(lon_deg)
    
        slat, clat = np.sin(lat), np.cos(lat)
        slon, clon = np.sin(lon), np.cos(lon)
    
        return np.array([
            [-slon,           -slat*clon,  clat*clon],
            [ clon,           -slat*slon,  clat*slon],
            [ 0.0,             clat,        slat     ]
        ])

    def antennas_offset_to_ECEF(self, TART_ECEF, TART_latlonalt, antenna_offsets):
        R = self.enu_to_ecef_matrix(TART_latlonalt[1], TART_latlonalt[0])
        antennas_ecef = TART_ECEF + (R @ antenna_offsets.T).T
        return antennas_ecef

    def calculate_elevation(self, sat_pos, site_pos):
        """Calculate elevation angle from site to satellite."""
        range_vec = np.array(sat_pos) - np.array(site_pos)
        up_vec = np.array(site_pos) / np.linalg.norm(site_pos)
        elevation_rad = np.arcsin(np.dot(range_vec, up_vec) / np.linalg.norm(range_vec))
        return np.degrees(elevation_rad)