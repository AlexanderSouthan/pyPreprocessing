#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 22:01:43 2020

@author: almami
"""

import numpy as np
import matplotlib.pyplot as plt

from pyPreprocessing.baseline_correction import generate_baseline, derivative
from pyPreprocessing.smoothing import smoothing
from pyRegression.nonlinear_regression import calc_function


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
                             baseline_parameters=[5, 0.01, 0.00003], noise_level=1)

baseline_ALSS = np.squeeze(
    generate_baseline(
        spectrum[1][np.newaxis], 'ALSS', smoothing=True))
baseline_iALSS = np.squeeze(
    generate_baseline(
        spectrum[1][np.newaxis], 'iALSS', smoothing=True))
baseline_drPLS = np.squeeze(
    generate_baseline(
        spectrum[1][np.newaxis], 'drPLS', smoothing=True))
baseline_SNIP = np.squeeze(
    generate_baseline(
        spectrum[1][np.newaxis], 'SNIP', smoothing=True, transform=False))
baseline_ModPoly = np.squeeze(
    generate_baseline(
        spectrum[1][np.newaxis], 'ModPoly', smoothing=True,
        wavenumbers=spectrum[0]))
baseline_IModPoly = np.squeeze(
    generate_baseline(
        spectrum[1][np.newaxis], 'IModPoly', smoothing=True,
        wavenumbers=spectrum[0]))
baseline_convex_hull = np.squeeze(
    generate_baseline(
        spectrum[1][np.newaxis], 'convex_hull', smoothing=True,
        wavenumbers=spectrum[0]))
baseline_PPF = np.squeeze(
    generate_baseline(
        spectrum[1][np.newaxis], 'PPF', smoothing=True,
        wavenumbers=spectrum[0], slope_threshold=0.07, step_threshold=0.01,
        check_point_number=50, poly_order=5))

plt.figure()
plt.plot(spectrum[0], spectrum[1])
plt.plot(spectrum[0], baseline_ALSS, label='ALSS')
plt.plot(spectrum[0], baseline_iALSS, label='iALLS')
plt.plot(spectrum[0], baseline_drPLS, label='drPLS')
plt.plot(spectrum[0], baseline_SNIP, label='SNIP')
plt.plot(spectrum[0], baseline_ModPoly, label='ModPoly')
plt.plot(spectrum[0], baseline_IModPoly, label='IModPoly')
plt.plot(spectrum[0], baseline_convex_hull, label='Convex hull')
plt.plot(spectrum[0], baseline_PPF, label='PPF')
# plt.plot(spectrum[0], np.squeeze(smoothing(spectrum[1][np.newaxis], 'sav_gol', savgol_points=19)))
plt.legend()

# plt.figure()
# plt.plot(spectrum[0], np.squeeze(derivative(spectrum[0], spectrum[1][np.newaxis])))

