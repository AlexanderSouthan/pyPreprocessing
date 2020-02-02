# -*- coding: utf-8 -*-
"""
Provides functions for smoothing of data rows oganized in 2D numpy arrays. 
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def smoothing(raw_data, mode, **kwargs):
    """
    Smoothes data rows with different algorithms.

    Parameters
    ----------
    raw_data : ndarray
        2D numpy array with the shape (N,M) containing N data rows to be
        smoothed. Each data row is represented by row in numpy array and
        contains M values. If only one data row is present, raw_data has the
        shape (1,M).
    mode : str
        Algorithm used for smoothing. Allowed modes are 'sav_gol' for Savitzky-
        Golay, 'rolling_median' for a median filter, 'pca' for smoothing based
        on principal component analysis(currently not functional).
    **kwargs for different modes
        sav_gol:
            deriv: int
                Derivative order to be calculated. Default is 0 (no
                derivative).
            savgol_points: int
                Number of point defining one side of the Savitzky-Golay window.
                Total window is 2*savgol_points+1. Default is 9.
            poly_order: int
                Polynomial order used for polynomial fitting of the Savitzky-
                Golay window. Default is 2.
        rolling_median:
            window: int
                Data points included in rolling window used for median
                calculations. Default is 5.
        pca:
            WILL BE ADDED AS SOON IT IS FUNCTIONAL

    Returns
    -------
    ndarray
        2D numpy array containing the smoothed data in the same shape as
        raw_data.

    """
    if mode == 'sav_gol':
        deriv = kwargs.get('deriv', 0)
        savgol_points = kwargs.get('savgol_points', 9)
        poly_order = kwargs.get('poly_order', 2)

        sav_gol_filtered_data = savgol_filter(raw_data, 1+2*savgol_points,
                                              poly_order, deriv=deriv,
                                              axis=1)
        return sav_gol_filtered_data

    if mode == 'rolling_median':  # reduces dataset by edge_value_count*2
        window = kwargs.get('window', 5)
        # next line due to pandas rolling window, look for numpy solution
        raw_data = pd.DataFrame(raw_data)

        edge_value_count = int((window-1)/2)
        median_filtered_data = raw_data.rolling(
                window, axis=1, center=True).median().iloc[
                :, edge_value_count:-edge_value_count]

        median_filtered_data = pd.concat(
            [raw_data.iloc[:, 0:2], median_filtered_data,
             raw_data.iloc[:, -3:-1]], axis=1)

        return median_filtered_data.values

    # not functional at the moment, needs pca function from spectroscopic data
    if mode == 'pca':
        pca_components = kwargs.get('pca_components', 3)

        pca_results = self.principal_component_analysis(
                pca_components, active_spectra=active_spectra)
        # pca_result is also calculated in multi2monochrome,
        # possibly it can be bundled in one place

        reconstructed_pca_image = pd.DataFrame(
            np.dot(pca_results['scores'], pca_results['loadings'])
            + self.mean_spectrum(active_spectra=active_spectra).values,
            index=active_spectra.index, columns=active_spectra.columns)

        return reconstructed_pca_image
