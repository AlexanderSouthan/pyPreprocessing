# -*- coding: utf-8 -*-
"""
Provides functions for smoothing of data rows oganized in 2D numpy arrays.
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA


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
        on principal component analysis.
    **kwargs for different modes
        sav_gol:
            deriv : int
                Derivative order to be calculated. Default is 0 (no
                derivative).
            savgol_points : int
                Number of point defining one side of the Savitzky-Golay window.
                Total window is 2*savgol_points+1. Default is 9.
            poly_order : int
                Polynomial order used for polynomial fitting of the Savitzky-
                Golay window. Default is 2.
        rolling_median:
            window: int
                Data points included in rolling window used for median
                calculations. Default is 5.
        pca:
            pca_components : int
                Number of principal components used to reconstruct the original
                data. Default is 5.

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

        smoothed_data = savgol_filter(raw_data, 1+2*savgol_points, poly_order,
                                      deriv=deriv, axis=1)

    if mode == 'rolling_median':  # reduces dataset by edge_value_count*2
        window = kwargs.get('window', 5)
        # next line due to pandas rolling window, look for numpy solution
        raw_data = pd.DataFrame(raw_data)

        edge_value_count = int((window-1)/2)
        smoothed_data = raw_data.rolling(
                window, axis=1, center=True).median().iloc[
                :, edge_value_count:-edge_value_count]

        # on the data edges, the original data is used, so the edges are not
        # smoothed. 
        smoothed_data = pd.concat(
            [raw_data.iloc[:, 0:edge_value_count], smoothed_data,
             raw_data.iloc[:, -1-edge_value_count:-1]], axis=1).values

    if mode == 'pca':
        pca_components = kwargs.get('pca_components', 5)

        pca = PCA(n_components=pca_components)
        scores = pca.fit_transform(raw_data)
        loadings = pca.components_

        smoothed_data = (
            np.dot(scores, loadings) + np.mean(raw_data, axis=0))
    
    return smoothed_data
