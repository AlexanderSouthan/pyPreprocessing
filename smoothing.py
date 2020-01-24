# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 16:58:12 2020

@author: Snijderfrey
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

def smoothing(raw_data, mode, **kwargs): 
    if mode == 'sav_gol':
        deriv = kwargs.get('deriv', 0)
        savgol_points = kwargs.get('savgol_points', 9)
        poly_order = kwargs.get('poly_order', 2)

        sav_gol_filtered_data = savgol_filter(raw_data, 1+2*savgol_points,
                                              poly_order, deriv=deriv,
                                              axis=1)
        return sav_gol_filtered_data

    if mode == 'rolling_median':  # reduces dataset by edge_value_count*2
        window = kwargs.get('window',5)
        raw_data = pd.DataFrame(raw_data)  # due to rolling window, look for numpy solution

        edge_value_count = int((window-1)/2)
        median_filtered_data = raw_data.rolling(
                window, axis=1, center=True).median().iloc[
                :, edge_value_count:-edge_value_count]
    
        median_filtered_data = pd.concat([raw_data.iloc[:, 0:2], median_filtered_data, raw_data.iloc[:, -3:-1]], axis=1)

        return median_filtered_data.values

    if mode == 'pca':  # not functional at the moment, needs pca function from spectroscopic data
        pca_components = kwargss.get('pca_components', 3)

        pca_results = self.principal_component_analysis(
                pca_components, active_spectra=active_spectra)
        # pca_result is also calculated in multi2monochrome,
        # possibly it can be bundled in one place

        reconstructed_pca_image = pd.DataFrame(
            np.dot(pca_results['scores'], pca_results['loadings'])
            + self.mean_spectrum(active_spectra=active_spectra).values,
            index=active_spectra.index, columns=active_spectra.columns)

        return reconstructed_pca_image