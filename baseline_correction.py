# -*- coding: utf-8 -*-
"""
Provides functions correct_baseline and generate_baseline which can be used for
baseline preprocessing of spectral data. See function docstrings for more
detail.

@author: Snijderfrey

to do:
    - Für ein einzelnes Spektrum testen
    - rubberband durchtesten (konkave und konvexe Spektren testen, Spektren mit großer Bandbreite an Intensitäten
      kokave Spektren evtl. durch concave hull: https://pdfs.semanticscholar.org/2397/17005c3ebd5d6a42fc833daf97a0edee1ce4.pdf
      und https://towardsdatascience.com/the-concave-hull-c649795c0f0f)
    - Test methods for ascending and descending wavenumbers
    - Negative Peaks berücksichtigen
    - Konvergenzkriterium für alle hinzufügen

"""

import numpy as np
from tqdm import tqdm
from scipy.signal import savgol_filter
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.spatial import ConvexHull


def correct_baseline(raw_data, mode, smoothing=True, transform=False,
                     **kwargs):
    """
    Calculates baseline data for raw_data with generate_baseline(...).
    Takes the same arguments like generate_baseline, for details see
    docstring of generate_baseline.
    """
    return raw_data - generate_baseline(raw_data, mode, smoothing=smoothing,
                                        transform=transform, **kwargs)


def generate_baseline(raw_data, mode, smoothing=True, transform=False,
                      **kwargs):
    """
    Calculates and returns baseline data on input datasets with different
    algorithms.

    Input data:
    -----------
    raw_data: ndarray
        Numpy 2D array of shape (N, M) with N datasets and M data points per
        dataset. If only one dataset is given, it has to have the shape (1, M).
    mode: str
        Algorithm for baseline calculation. Allowed values:
        'convex_hull', 'ALSS', 'iALSS', 'drPLS', 'SNIP', 'ModPoly', 'IModPoly',
        'PPF'.
    smoothing: bool
        True if datasets should be smoothed before calculation (recommended),
        otherwise False.
    transform: bool
        True if datasets should be transformed before calculation,
        otherwise False.

    kwargs for smoothing == True
    ---------------------------
    savgol_window: int
        window size for Savitzky-Golay window, default=19.
    savgol_order: int
        polynomial order for Savitzky-Golay filter, default=2.

    kwargs for different baseline modes:
    ------------------------------------
    convex_hull:
        wavenumbers: ndarray
            Numpy array containing wavenumbers or wavelengths of datasets.
            Must have M elements. default=np.arange(M)
    ALSS:
        lam: float
            default=10000
        p: float
            default=0.001
        n_iter: int
            default=10
        conv_crit: float
            default=0.001
    iALSS:
        lam: float
            default=2000
        lam_1: float
            default=0.01
        p: float
            default=0.01
        n_iter: int
            default=10
        conv_crit: float
            default=0.001
        wavenumbers: ndarray
            Numpy array containing wavenumbers or wavelengths of datasets.
            Must have M elements. default=np.arange(M)
    drPLS:
        lam: float
            default=1000000
        eta: float
            default=0.5
        n_iter: int
            default=100
        conv_crit: float
            default=0.001
    SNIP:
        n_iter: int
            default=100
    ModPoly, IModPoly:
        wavenumbers: ndarray
            Numpy array containing wavenumbers or wavelengths of datasets.
            Must have M elements. default=np.arange(M)
        n_iter: int
            default=100
        poly_order: int
            default=5
    PPF
    
    """

    if smoothing:
        savgol_window = kwargs.get('savgol_window', 19)
        savgol_order = kwargs.get('savol_order', 2)
        raw_data = np.around(savgol_filter(raw_data, savgol_window,
                                           savgol_order, deriv=0, axis=1),
                             decimals=6)
    if transform:  # currently LLS transformation
        spectra_minimum_value = np.min(raw_data)
        raw_data = raw_data - spectra_minimum_value
        raw_data = np.log(np.log(np.sqrt(raw_data + 1)+1)+1)

    baseline_data = np.zeros_like(raw_data)
    baseline_modes = ['convex_hull', 'ALSS', 'iALSS', 'drPLS', 'SNIP',
                      'ModPoly', 'IModPoly','PPF']

    if mode == baseline_modes[0]:  # convex_hull
        # based on (but improved a bit)
        # https://dsp.stackexchange.com/questions/2725/how-to-perform-a-rubberband-correction-on-spectroscopic-data

        # set mode specific parameters
        wavenumbers = kwargs.get('wavenumbers', np.arange(raw_data.shape[1]))
        #############################

        for ii, current_spectrum in enumerate(tqdm(raw_data)):
            hull_vertices = ConvexHull(
                np.array(list(zip(wavenumbers, current_spectrum)))).vertices
            # Rotate convex hull vertices until they start from the lowest one
            hull_vertices = np.roll(hull_vertices, -np.argmin(hull_vertices))
            # split vertices into upper and lower part
            hull_vertices_section_1 = hull_vertices[:np.argmax(hull_vertices)+1]
            hull_vertices_section_2 = np.sort(np.insert(hull_vertices[np.argmax(hull_vertices):], 0, hull_vertices[0]))
            # calculate spectrum mean intensities of upper and lower vertices
            raw_mean_1 = np.mean(current_spectrum[hull_vertices_section_1])
            raw_mean_2 = np.mean(current_spectrum[hull_vertices_section_2])

            # Select lower vertices as baseline vertices
            if raw_mean_1 > raw_mean_2:
                baseline_vertices = hull_vertices_section_2
            else:
                baseline_vertices = hull_vertices_section_1

            # Create baseline using linear interpolation between vertices
            baseline_data[ii, :] = np.interp(wavenumbers, np.flip(wavenumbers[baseline_vertices]), np.flip(current_spectrum[baseline_vertices]))

    elif mode == baseline_modes[1]:  # ALSS
        # according to
        # "Baseline Correction with Asymmetric Least Squares Smoothing"
        # by P. Eilers and H. Boelens.
        # https://zanran_storage.s3.amazonaws.com/www.science.uva.nl/ContentPages/443199618.pdf

        # set mode specific parameters
        lam = kwargs.get('lam', 10000)
        p = kwargs.get('p', 0.001)
        n_iter = kwargs.get('n_iter', 10)
        conv_crit = kwargs.get('conv_crit', 0.001)
        #############################

        L = raw_data.shape[1]
        D = diags([1, -2, 1], [0, -1, -2], shape=(L, L-2), format='csr')
        D = D.dot(D.transpose())

        for ii, current_spectrum in enumerate(tqdm(raw_data)):

            # this is the code for the fitting procedure
            w = np.ones(L)
            W = diags(w, format='csr')
            z = w

            for jj in range(int(n_iter)):
                W.setdiag(w)
                Z = W + lam * D
                z_prev = z
                z = spsolve(Z, w*current_spectrum, permc_spec='NATURAL')
                if np.linalg.norm(z - z_prev) > conv_crit:
                    w = p * (current_spectrum > z) + (1-p) * (current_spectrum < z)
                else:
                    break
            # end of fitting procedure

            baseline_data[ii, :] = z

    elif mode == baseline_modes[2]:  # iALSS
        # according to "Anal. Methods, 2014, 6, 4402–4407."

        # set mode specific parameters
        lam = kwargs.get('lam', 2000)
        lam_1 = kwargs.get('lam_1', 0.01)
        p = kwargs.get('p', 0.01)
        n_iter = kwargs.get('n_iter', 10)
        conv_crit = kwargs.get('conv_crit', 0.001)
        wavenumbers = kwargs.get('wavenumbers', np.arange(raw_data.shape[1]))
        #############################

        L = raw_data.shape[1]
        fit_coeffs = np.polynomial.polynomial.polyfit(wavenumbers,
                                                      raw_data.T, 2)
        w_start_all = np.polynomial.polynomial.polyval(wavenumbers, fit_coeffs)

        D = diags([1, -2, 1], [0, -1, -2], shape=(L, L-2), format='csr')
        D = D.dot(D.transpose())
        D_1 = diags([-1, 1], [0, -1], shape=(L, L-1), format='csr')
        D_1 = D_1.dot(D_1.transpose())

        for ii, current_spectrum in enumerate(tqdm(raw_data)):

            # this is the code for the fitting procedure
            w = w_start_all[ii, :]
            z = w
            W = diags(w, format='csr')
            w = p * (current_spectrum > z) + (1-p) * (current_spectrum < z)

            for jj in range(int(n_iter)):
                W.setdiag(w)
                W = W.dot(W.transpose())
                Z = W + lam_1 * D_1 + lam * D
                R = (W + lam_1 * D_1) * current_spectrum
                z_prev = z
                z = spsolve(Z, R, permc_spec='NATURAL')
                if np.linalg.norm(z - z_prev) > conv_crit:
                    w = p * (current_spectrum > z) + (1-p) * (current_spectrum < z)
                else:
                    break
            # end of fitting procedure

            baseline_data[ii, :] = z

    elif mode == baseline_modes[3]:  # drPLS
        # according to "Applied Optics, 2019, 58, 3913-3920."

        # set mode specific parameters
        lam = kwargs.get('lam', 1000000)
        eta = kwargs.get('eta', 0.5)
        n_iter = kwargs.get('n_iter', 100)
        conv_crit = kwargs.get('conv_crit', 0.001)
        #############################

        L = raw_data.shape[1]

        D = diags([1, -2, 1], [0, -1, -2], shape=(L, L-2), format='csr')
        D = D.dot(D.transpose())
        D_1 = diags([-1, 1], [0, -1], shape=(L, L-1), format='csr')
        D_1 = D_1.dot(D_1.transpose())

        w_0 = np.ones(L)
        I_n = diags(w_0, format='csr')

        for ii, current_spectrum in enumerate(tqdm(raw_data)):

            # this is the code for the fitting procedure
            w = w_0
            W = diags(w, format='csr')
            Z = w_0

            for jj in range(int(n_iter)):
                W.setdiag(w)
                Z_prev = Z
                Z = spsolve(W + D_1 + lam * (I_n - eta*W) *
                            D, W*current_spectrum, permc_spec='NATURAL')
                if np.linalg.norm(Z - Z_prev) > conv_crit:
                    d = current_spectrum - Z
                    d_negative = d[d < 0]
                    sigma_negative = np.std(d_negative)
                    mean_negative = np.mean(d_negative)
                    w = 0.5 * (1 - np.exp(jj) * (d - (-mean_negative + 2*sigma_negative))/sigma_negative / (1 + np.abs(np.exp(jj) * (d - (-mean_negative + 2*sigma_negative))/sigma_negative)))
                else:
                    break
            # end of fitting procedure

            baseline_data[ii, :] = Z

    elif mode == baseline_modes[4]:  # SNIP
        # according to "Nuclear Instruments and Methods in Physics Research
        # 934 (1988) 396-402."
        # and Nuclear Instruments and Methods in Physics Research Section A:
        # Accelerators, Spectrometers, Detectors and Associated Equipment 1997,
        # 401 (1), 113-132

        # set mode specific parameters
        n_iter = kwargs.get('n_iter', 100)
        #############################

        spectrum_points = raw_data.shape[1]
        working_spectra = np.zeros_like(raw_data)

        for pp in tqdm(np.arange(1, n_iter+1)):
            r1 = raw_data[:, pp:spectrum_points-pp]
            r2 = (np.roll(raw_data, -pp, axis=1)[:, pp:spectrum_points-pp] +
                  np.roll(raw_data, pp, axis=1)[:, pp:spectrum_points-pp])/2
            working_spectra = np.minimum(r1, r2)
            raw_data[:, pp:spectrum_points-pp] = working_spectra

        baseline_data = raw_data

    elif mode in baseline_modes[5:7]:  # ModPoly, IModPoly
        # according to Applied Spectroscopy, 2007, 61 (11), 1225-1232.
        # without dev: Chemometrics and Intelligent Laboratory Systems 82 (2006) 59– 65.
        #               Maybe also ModPoly from first source?

        # set mode specific parameters
        wavenumbers = kwargs.get('wavenumbers', np.arange(raw_data.shape[1]))
        n_iter = kwargs.get('n_iter', 100)
        poly_order = kwargs.get('poly_order', 5)
        #############################

        wavenumbers_start = wavenumbers
        # previous_dev = 0

        for ii, current_spectrum in enumerate(tqdm(raw_data)):
            wavenumbers = wavenumbers_start
            for jj in range(int(n_iter)):
                fit_coeffs = np.polynomial.polynomial.polyfit(wavenumbers,
                                                              current_spectrum,
                                                              poly_order)
                fit_data = np.polynomial.polynomial.polyval(wavenumbers,
                                                            fit_coeffs)

                if mode == baseline_modes[5]:  # ModPoly
                    dev = 0
                else:  #IModPoly
                    residual = current_spectrum - fit_data
                    dev = residual.std()
                    # if abs((dev - previous_dev)/dev) < 0.01:
                    #    break

                if jj == 0:
                    mask = (current_spectrum <= fit_data + dev)
                    wavenumbers = wavenumbers[mask]
                    current_spectrum = current_spectrum[mask]
                    fit_data = fit_data[mask]
                np.copyto(current_spectrum, fit_data + dev,
                          where=(current_spectrum >= (fit_data+dev)))
                # previous_dev = dev

            baseline_data[ii, :] = np.polynomial.polynomial.polyval(
                wavenumbers_start, fit_coeffs)
            
    elif mode == baseline_modes[7]:  # PPF
        # according to Photonic Sensors 2018, 8(4), 332-340.

        # set mode specific parameters
        # wavenumbers = kwargs.get('wavenumbers', np.arange(raw_data.shape[1]))
        # n_iter = kwargs.get('n_iter', 100)
        savgol_window_deriv = 19
        savgol_order_deriv = 2
        slope_threshold = 5
        segment_points = 20
        poly_order = kwargs.get('poly_order', 5)
        #############################
        raw_data_derivative = np.around(
                savgol_filter(raw_data, savgol_window_deriv,
                              savgol_order_deriv, deriv=1, axis=1), decimals=6)
        derivative_sign_changes = np.diff(
                np.sign(raw_data_derivative), axis=1,
                append=raw_data_derivative[:, -1, np.newaxis])
        # peak_maxima = (derivative_sign_changes == -2)
        peak_boundaries = (derivative_sign_changes == 2)
#        check_points = np.roll(peak_boundaries, segment_points, axis=1)

        deriv_diffs = raw_data_derivative - np.roll(raw_data_derivative, segment_points, axis=1)
        deriv_diffs_at_peak_bounds = np.abs(deriv_diffs * peak_boundaries)  # np.roll(peak_boundaries, segment_points, axis=1))
        segment_points = (deriv_diffs_at_peak_bounds < slope_threshold) & (deriv_diffs_at_peak_bounds > 0)
        
        
        return segment_points

    else:
        raise ValueError('No valid baseline mode entered. Allowed modes are{0}'.format(baseline_modes))

    if transform:
        baseline_data = (np.exp(np.exp(raw_data)-1)-1)**2 - 1 + spectra_minimum_value
    return np.around(baseline_data, decimals=6)
