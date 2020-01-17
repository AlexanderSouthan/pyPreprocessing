# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 22:15:37 2019

@author: AlMaMi

to do:
    - Für ein einzelnes Spektrum testen
    - rubberband durchtesten (konkave und konvexe Spektren testen, Spektren mit großer Bandbreite an Intensitäten
      kokave Spektren evtl. durch concave hull: https://pdfs.semanticscholar.org/2397/17005c3ebd5d6a42fc833daf97a0edee1ce4.pdf
      und https://towardsdatascience.com/the-concave-hull-c649795c0f0f)
    - Test methods for ascending and descending wavenumbers

"""

import numpy as np
from tqdm import tqdm
from scipy.signal import savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import ConvexHull


def rubberband_baseline(raw_data, wavenumbers, smoothing=True):
    """based on (but improved a lot)
    https://dsp.stackexchange.com/questions/2725/how-to-perform-a-rubberband-correction-on-spectroscopic-data
    """

    if smoothing:
        raw_data = np.around(savgol_filter(raw_data, 19, 2, deriv=0, axis=1),
                             decimals=6)

    rubberband_baseline_data = np.empty_like(raw_data)

    for ii, current_spectrum in enumerate(tqdm(raw_data)):
        hull_vertices = ConvexHull(np.array(list(zip(wavenumbers, current_spectrum)))).vertices
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
            correct_vertices = hull_vertices_section_2
        else:
            correct_vertices = hull_vertices_section_1
        
        # Create baseline using linear interpolation between vertices
        rubberband_baseline_data[ii, :] = np.interp(wavenumbers, np.flip(wavenumbers[correct_vertices]), np.flip(current_spectrum[correct_vertices]))

    return np.around(rubberband_baseline_data, decimals=6)


def ALSS_baseline(raw_data, lam, p, n_iter, conv_crit=0.001, smoothing=True):
    """according to
    \"Baseline Correction with Asymmetric Least Squares Smoothing\"
    by P. Eilers and H. Boelens.
    https://zanran_storage.s3.amazonaws.com/www.science.uva.nl/ContentPages/443199618.pdf"""

    if smoothing:
        raw_data = np.around(savgol_filter(raw_data, 19, 2, deriv=0, axis=1),
                             decimals=6)

    L = raw_data.shape[1]
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2), format='csr')
    D = D.dot(D.transpose())

    ALSS_baseline_data = np.empty_like(raw_data)
    for ii, current_spectrum in enumerate(tqdm(raw_data)):

        # this is the code for the fitting procedure
        w = np.ones(L)
        W = sparse.diags(w, format='csr')
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

        ALSS_baseline_data[ii, :] = z

    return np.around(ALSS_baseline_data, decimals=6)


def iALSS_baseline(raw_data, wavenumbers, lam, lam_1, p, n_iter, conv_crit=0.001, smoothing=True):
    """according to \"Anal. Methods, 2014, 6, 4402–4407.\""""

    if smoothing:
        raw_data = np.around(savgol_filter(raw_data, 19, 2, deriv=0, axis=1), decimals=6)

    L = raw_data.shape[1]
    fit_coeffs = np.polynomial.polynomial.polyfit(wavenumbers, raw_data.T, 2)
    w_start_all = np.polynomial.polynomial.polyval(wavenumbers, fit_coeffs)

    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2), format='csr')
    D = D.dot(D.transpose())
    D_1 = sparse.diags([-1, 1], [0, -1], shape=(L, L-1), format='csr')
    D_1 = D_1.dot(D_1.transpose())

    iALSS_baseline_data = np.empty_like(raw_data)
    for ii, current_spectrum in enumerate(tqdm(raw_data)):

        # this is the code for the fitting procedure
        w = w_start_all[ii, :]
        z = w
        W = sparse.diags(w, format='csr')
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

        iALSS_baseline_data[ii, :] = z

    return np.around(iALSS_baseline_data, decimals=6)


def drPLS_baseline(raw_data, lam, eta, n_iter, conv_crit=0.001, smoothing=True):
    """according to \"Applied Optics, 2019, 58, 3913-3920.\""""

    if smoothing:
        raw_data = np.around(savgol_filter(raw_data, 19, 2, deriv=0, axis=1), decimals=6)

    L = raw_data.shape[1]

    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2), format='csr')
    D = D.dot(D.transpose())
    D_1 = sparse.diags([-1, 1], [0, -1], shape=(L, L-1), format='csr')
    D_1 = D_1.dot(D_1.transpose())

    w_0 = np.ones(L)
    I_n = sparse.diags(w_0, format='csr')

    drPLS_baseline_data = np.empty_like(raw_data)
    for ii, current_spectrum in enumerate(tqdm(raw_data)):

        # this is the code for the fitting procedure
        w = w_0
        W = sparse.diags(w, format='csr')
        Z = w_0

        for jj in range(int(n_iter)):
            W.setdiag(w)
            Z_prev = Z
            Z = spsolve(W + D_1 + lam * (I_n - eta*W) * D, W*current_spectrum, permc_spec='NATURAL')
            if np.linalg.norm(Z - Z_prev) > conv_crit:
                d = current_spectrum - Z
                d_negative = d[d < 0]
                sigma_negative = np.std(d_negative)
                mean_negative = np.mean(d_negative)
                w = 0.5 * (1 - np.exp(jj) * (d - (-mean_negative + 2*sigma_negative))/sigma_negative / (1 + np.abs(np.exp(jj) * (d - (-mean_negative + 2*sigma_negative))/sigma_negative)))
            else:
                break
        # end of fitting procedure

        drPLS_baseline_data[ii, :] = Z

    return np.around(drPLS_baseline_data, decimals=6)


def SNIP_baseline(raw_data, n_iter, smoothing=True):
    """according to \"Nuclear Instruments and Methods in Physics Research 934 (1988) 396-402.\""""

    if smoothing:
        raw_data = np.around(savgol_filter(raw_data, 19, 2, deriv=0, axis=1), decimals=6)

    spectra_minimum_value = np.min(raw_data)

    raw_data_spectral_part = raw_data - spectra_minimum_value
    spectrum_points = raw_data_spectral_part.shape[1]
    raw_data_transformed = np.log(np.log(np.sqrt(raw_data_spectral_part + 1)+1)+1)

    working_spectra = np.zeros_like(raw_data_spectral_part)

    for pp in tqdm(np.arange(1, n_iter+1)):
        r1 = raw_data_transformed[:, pp:spectrum_points-pp]
        r2 = (np.roll(raw_data_transformed, -pp,axis=1)[:, pp:spectrum_points-pp] + np.roll(raw_data_transformed, pp,axis=1)[:, pp:spectrum_points-pp])/2
        working_spectra = np.minimum(r1, r2)
        raw_data_transformed[:, pp:spectrum_points-pp] = working_spectra

    return np.around((np.exp(np.exp(raw_data_transformed)-1)-1)**2 - 1 + spectra_minimum_value, decimals=6)
