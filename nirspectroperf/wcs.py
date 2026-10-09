"""
This module is for modeling the mapping from pixel to world coordinates in the 
spectrograph slits. 

Note this may not be necessary for WCSs that are linear with the sky/pixel offsets (like NIRSpec)
"""
from abc import ABC, abstractmethod

import numpy as np


class W2P(ABC):
    """
    A class that maps from a specific slit-centric view of world coordinates to pixel coordinates.
    """
    @abstractmethod
    def evaluate_wcs_values(self, ra, dec, wave):
        """
        ra, dec should be in degrees, wave should be in microns. 
        
        Returns xslit, yslit in pixels
        """
        pass

class WCSW2P(W2P):
    def __init__(self, wcs):
        self.wcs = wcs

    def evaluate_wcs_values(self, ra, dec, wave):
        if not isinstance(ra, np.ndarray):
            ra = np.array([ra])
        if not isinstance(dec, np.ndarray):
            dec = np.array([dec])
        if not isinstance(wave, np.ndarray):
            wave = np.array([wave])
        return self.wcs.world_to_pixel_values(ra, dec, wave)