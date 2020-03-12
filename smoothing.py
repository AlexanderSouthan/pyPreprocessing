# -*- coding: utf-8 -*-
"""
Provides functions for smoothing of data rows oganized in 2D numpy arrays.
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA


def smoothing(raw_data, mode, interpolate=False, point_mirror=True, **kwargs):
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
    interpolate : boolean
        False if x coordinate is evenly spaced. True if x coordinate is not
        evenly spaced, then raw_data is interpolated to an evenly spaced
        x coordinate. default=False
    point_mirror : boolean
        Dataset is point reflected at both end points before smoothing to
        reduce artifacts at the data edges.
    **kwargs for interpolate=True
        x_coordinate : ndarray
            1D numpy array with shape (M,) used for interpolation.
        data_points : int
            number of data points returned after interpolation. Default is one
            order of magnitude more than M.
    **kwargs for different smoothing modes
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
            savgol_mode : str
                Must be ‘mirror’, ‘constant’, ‘nearest’, ‘wrap’ or ‘interp’.
                See documentation of scipy.signal.savgol_filter.
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
        raw_data if interpolate is false. Else tuple containing interpolated
        x coordinates and 2D numpy array in the shape of
        (N,10**np.ceil(np.log10(len(x_coordinate)))).

    """

    # Preprocessing of input data for unevenly spaced x coordinate
    if interpolate:
        x_coordinate = kwargs.get('x_coordinate', np.linspace(
            0, 1000, raw_data.shape[1]))
        data_points = kwargs.get('data_points',
                                 int(10**np.ceil(np.log10(len(x_coordinate)))))

        itp = interp1d(x_coordinate, raw_data, kind='linear')
        x_interpolated = np.linspace(x_coordinate[0], x_coordinate[-1],
                                     data_points)
        raw_data = itp(x_interpolated)

    # Optional extension of smoothed data by point mirrored raw data.
    if point_mirror:
        raw_data = np.concatenate(
            ((-np.flip(raw_data, axis=1)+2*raw_data[:, 0, np.newaxis])[:, :-1],
             raw_data, (-np.flip(raw_data, axis=1) +
                        2*raw_data[:, -1, np.newaxis])[:, 1:]), axis=1)
        #raw_data = np.concatenate((-np.squeeze(raw_data.T)[::-1]+2*np.squeeze(raw_data.T)[0],np.squeeze(raw_data.T),-np.squeeze(raw_data.T)[::-1]+2*np.squeeze(raw_data.T)[-1]))[np.newaxis]

    smoothing_modes = ['sav_gol', 'rolling_median', 'pca']

    if mode == smoothing_modes[0]:  # sav_gol
        deriv = kwargs.get('deriv', 0)
        savgol_points = kwargs.get('savgol_points', 9)
        poly_order = kwargs.get('poly_order', 2)
        savgol_mode = kwargs.get('savgol_mode', 'nearest')

        smoothed_data = savgol_filter(raw_data, 1+2*savgol_points, poly_order,
                                      deriv=deriv, axis=1, mode=savgol_mode)

    elif mode == smoothing_modes[1]:  # rolling_median
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

    elif mode == smoothing_modes[2]:  # pca
        pca_components = kwargs.get('pca_components', 5)

        pca = PCA(n_components=pca_components)
        scores = pca.fit_transform(raw_data)
        loadings = pca.components_

        smoothed_data = (
            np.dot(scores, loadings) + np.mean(raw_data, axis=0))

    else:
        raise ValueError('No valid baseline mode entered. Allowed modes are '
                         '{0}'.format(smoothing_modes))

    # Removal of previously added point mirrored data.
    if point_mirror:
        #smoothed_data = np.split(np.squeeze(smoothed_data.T),3)[1][np.newaxis]
        smoothed_data = smoothed_data[
            :, int(np.ceil(smoothed_data.shape[1]/3)-1):
                int(2*np.ceil(smoothed_data.shape[1]/3)-1)]

    if interpolate:
        return (x_interpolated, smoothed_data)
    else:
        return smoothed_data
