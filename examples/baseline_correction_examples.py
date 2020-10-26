#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 22:01:43 2020

@author: almami
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyPreprocessing.baseline_correction import generate_baseline, derivative
from pyPreprocessing.smoothing import smoothing
from pyRegression.nonlinear_regression import calc_function
from skimage.filters import threshold_otsu, threshold_local, threshold_triangle
from skimage.feature import canny, blob_log, blob_doh, blob_dog
from skimage.restoration import estimate_sigma
from scipy.signal import find_peaks

def otsu(hist_x, hist_y):
    total = np.sum(hist_y)
    top = len(hist_y)
    sumB = 0
    wB = 0
    maximum = 0.0
    sum1 = np.dot(np.arange(top), hist_y)
    
    for ii in np.arange(top):
        wF = total -wB
        if (wB > 0) and (wF > 0):
            mF = (sum1 - sumB) / wF
            val = wB * wF * ((sumB / wB) - mF) * ((sumB / wB) - mF)
            if val >= maximum:
                level = ii
                maximum = val
        wB += hist_y[ii]
        sumB += ii * hist_y[ii]
    
    otsu_thresh = hist_x[level]
    
    return otsu_thresh


def simulate_spectrum(peak_centers, peak_amplitudes, peak_widths,
                      baseline_type='polynomial', baseline_parameters=[1],
                      noise_level=1, wn_start=0, wn_end=1000,
                      data_points=1000):
    """
    Calculate spectrum with Gaussian peaks.

    Parameters
    ----------
    peak_centers : list of float
        The peak centers.
    peak_amplitudes : list of float
        The peak amplitudes, i.e. the maximum value of the peak.
    peak_widths : list of float
        The sigma of the Gaussian paeks.
    baseline_type : str, optional
        The baseline type, currently only 'polynomial' using the calc_function
        polynomial calculation. The default is 'polynomial'.
    baseline_parameters : list of float, optional
        Parameters passed to calc_function. The default is [1], resulting in a
        constant baseline with a value of 1 in case of baseline_type is
        'polynomial'.
    noise_level : float, optional
        The maximum level of the noise. The default is 1.
    wn_start : float, optional
        The start wavenumber used for spectrum calculation. The default is 0.
    wn_end : float, optional
        The end wavenumber used for spectrum calculation. The default is 1000.
    data_points : int, optional
        The number of evenly spaced data points between wn_start and wn_end
        used for spectrum calculation. The default is 1000.

    Returns
    -------
    ndarray
        2D array with the wavenumbers and the intensities.

    """
    # Calculate wavennumbers
    wavenumbers = np.linspace(wn_start, wn_end, num=data_points)

    # Reshape Gaussian parameters and pass them to calc_function for pure
    # spectrum intensities without noise and baseline contributions
    gauss_parameters = np.stack(
        [peak_amplitudes, peak_centers, np.zeros_like(peak_amplitudes),
         peak_widths]).reshape((1, -1), order='F')
    pure_intensities = calc_function(wavenumbers, gauss_parameters, 'Gauss')

    # Calculate noise as random Gaussian noise
    rng = np.random.default_rng()
    noise = rng.standard_normal(len(pure_intensities)) * noise_level

    # Calculate baseline
    if baseline_type == 'polynomial':
        baseline = calc_function(wavenumbers, baseline_parameters,
                                 'polynomial')
    else:
        baseline = np.zeros_like(pure_intensities)

    # Calculate spectrum intensities as the sum of pure intensities, noise and
    # baseline contribution
    intensities = pure_intensities + noise + baseline

    return np.array([wavenumbers, intensities])


spectrum = simulate_spectrum([200, 250, 500], [10, 5, 20], [10, 40, 5],
                             baseline_parameters=[5, 0.01, 0.0003], noise_level=1)
spectrum_clean = simulate_spectrum([200, 250, 500], [10, 5, 20], [10, 40, 5],
                             baseline_parameters=[0], noise_level=0)

smoothed_spectrum = smoothing(spectrum[1][np.newaxis], 'sav_gol', savgol_points=10, savgol_order=9)
derived_spectrum = derivative(spectrum[0], smoothed_spectrum)
derived_spectrum_2 = derivative(spectrum[0], smoothed_spectrum, order=2)

# baseline_ALSS = np.squeeze(
#     generate_baseline(
#         spectrum[1][np.newaxis], 'ALSS', smoothing=True))
# baseline_iALSS = np.squeeze(
#     generate_baseline(
#         spectrum[1][np.newaxis], 'iALSS', smoothing=True))
# baseline_drPLS = np.squeeze(
#     generate_baseline(
#         spectrum[1][np.newaxis], 'drPLS', smoothing=True))
# baseline_SNIP = np.squeeze(
#     generate_baseline(
#         spectrum[1][np.newaxis], 'SNIP', smoothing=True, transform=False))
# baseline_ModPoly = np.squeeze(
#     generate_baseline(
#         spectrum[1][np.newaxis], 'ModPoly', smoothing=True,
#         wavenumbers=spectrum[0]))
baseline_IModPoly = np.squeeze(
    generate_baseline(
        spectrum[1][np.newaxis], 'IModPoly', smoothing=True,
        wavenumbers=spectrum[0], poly_order=3))
# baseline_convex_hull = np.squeeze(
#     generate_baseline(
#         spectrum[1][np.newaxis], 'convex_hull', smoothing=True,
#         wavenumbers=spectrum[0]))
# baseline_PPF = np.squeeze(
#     generate_baseline(
#         spectrum[1][np.newaxis], 'PPF', smoothing=True,
#         wavenumbers=spectrum[0], slope_threshold=0.07, step_threshold=0.01,
#         check_point_number=50, poly_order=5))

plt.figure()
plt.plot(spectrum[0], spectrum[1])
plt.plot(spectrum[0], smoothed_spectrum.T)
# plt.plot(spectrum[0], baseline_ALSS, label='ALSS')
# plt.plot(spectrum[0], baseline_iALSS, label='iALLS')
# plt.plot(spectrum[0], baseline_drPLS, label='drPLS')
# plt.plot(spectrum[0], baseline_SNIP, label='SNIP')
# plt.plot(spectrum[0], baseline_ModPoly, label='ModPoly')
plt.plot(spectrum[0], baseline_IModPoly, label='IModPoly')
# plt.plot(spectrum[0], baseline_convex_hull, label='Convex hull')
# plt.plot(spectrum[0], baseline_PPF, label='PPF')
# # plt.plot(spectrum[0], np.squeeze(smoothing(spectrum[1][np.newaxis], 'sav_gol', savgol_points=19)))
# plt.legend()

plt.figure()
plt.plot(spectrum[0], np.squeeze(derived_spectrum))

plt.figure()
plt.plot(spectrum[0], np.squeeze(derived_spectrum_2))

plt.figure()
plt.plot(spectrum[0], spectrum[1]-baseline_IModPoly.T)
plt.plot(spectrum[0], spectrum_clean[1])

# plt.figure()
# plt.hist(np.squeeze(np.abs(derived_spectrum)), bins=100)

# deriv_hist = np.histogram(np.squeeze(np.abs(derived_spectrum)), bins=100)
# hist_x = deriv_hist[1][:-1] + np.diff(deriv_hist[1])/2
# hist_y = deriv_hist[0]
# plt.bar(hist_x, hist_y, width=np.diff(deriv_hist[1]).mean())

# plt.figure()
# deriv_2_hist = np.histogram(np.squeeze(np.abs(derived_spectrum_2)), bins=100)
# hist_x_2 = deriv_2_hist[1][:-1] + np.diff(deriv_2_hist[1])/2
# hist_y_2 = deriv_2_hist[0]
# plt.bar(hist_x_2, hist_y_2, width=np.diff(deriv_2_hist[1]).mean())

# otsu_skimage = threshold_otsu(np.squeeze(np.abs(derived_spectrum)))

# otsu_own = otsu(hist_x, hist_y)

# thresh_local = threshold_local(np.abs(derived_spectrum), 21)
# thresh_triangle = threshold_triangle(np.abs(derived_spectrum), nbins=100)

# plt.figure()
# plt.plot(spectrum[0], np.squeeze(np.abs(derived_spectrum)))
# plt.axhline(thresh_triangle)

# can = canny(spectrum[1][np.newaxis], sigma=1)
# print('Summe', np.sum(can))

# blob_log_r = blob_log(spectrum[1][np.newaxis])
# blob_doh_r = blob_doh(spectrum[1][np.newaxis])
# blob_dog_r = blob_dog(spectrum[1][np.newaxis])

# peaks = find_peaks(spectrum[1])

# noise = estimate_sigma(spectrum[1][np.newaxis])

std_rolling = pd.Series(spectrum[1]).rolling(25, center=True).std()
mean_rolling = pd.Series(spectrum[1]).rolling(25, center=True).mean()

plt.figure()
plt.plot(spectrum[0], std_rolling*mean_rolling)