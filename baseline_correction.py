# -*- coding: utf-8 -*-
"""
Provides functions correct_baseline and generate_baseline which can be used for
baseline preprocessing of spectral data. See function docstrings for more
detail.

to do:
    - test convex_hull thorougly (concave and convex spectra, spectra with
                                  big intensity range...)
     - look into concave hull for concave spectra:
          https://pdfs.semanticscholar.org/2397/17005c3ebd5d6a42fc833daf97a0edee1ce4.pdf
      and https://towardsdatascience.com/the-concave-hull-c649795c0f0f)
    - Test methods for ascending and descending wavenumbers
    - Make usable for negative peaks
    - Add convergence criteria for all methods
    - Have a look at dx.doi.org/10.1039/c3an00743j and dx.doi.org/10.3390/s20072015 for alternative methods
"""

import numpy as np
from tqdm import tqdm
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.spatial import ConvexHull

from pyRegression.polynomial_regression import (polynomial_fit,
                                                piecewise_polynomial,
                                                piecewise_polynomial_fit)
from .transform import transform as transform_spectra
from .smoothing import smoothing as smooth_spectra


def correct_baseline(raw_data, mode, smoothing=True, transform=False,
                     **kwargs):
    """
    Calculate baseline data for raw_data with generate_baseline(...).

    Takes the same arguments like generate_baseline, for details see
    docstring of generate_baseline.
    """
    return raw_data - generate_baseline(raw_data, mode, smoothing=smoothing,
                                        transform=transform, **kwargs)


def generate_baseline(raw_data, mode, smoothing=True, transform=False,
                      **kwargs):
    """
    Calculate baseline data on input datasets with different algorithms.

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

    kwargs for transform == True
    ---------------------------
    currently none, but will to be added in future versions.

    kwargs for different baseline modes:
    ------------------------------------
    convex_hull:
        wavenumbers: ndarray
            Numpy array containing wavenumbers or wavelengths of datasets.
            Must have M elements and must be sorted. default=np.arange(M)
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
            Must have M elements and must be sorted. default=np.arange(M)
        n_iter: int
            default=100
        poly_order: int
            default=5
        fixed_points: list of tuples, optional
            Contains constraints for points that the baseline must
            pass through. Each point is given by a tuple of two numbers,
            the wavenumber and the intensity of the point. If no point
            constraints are to be applied, this must be None. The
            default is None.
        fixed_slopes: list of tuples, optional
            Contains constraints for slopes that the fit functions must
            have at specific wavenumbers. Each slope is given by a tuple of
            two numbers, the wavenumber and the slope. If no slope
            constraints are to be applied, this must be None. The
            default is None.
    PPF:
        wavenumbers: ndarray
            Numpy array containing wavenumbers or wavelengths of datasets.
            Must have M elements and must be sorted. default=np.arange(M)
        segment_borders : list of int or float, optional
            The values with respect to wavenumbers at which the data is divided
            into segments. An arbitrary number of segment borders may be given,
            but it is recommended to provide a sorted list in order to avoid
            confusion. If the list is not sorted, it will be sorted. The
            default is [wavenumbers[len(wavenumbers)//2]], resulting in a
            segmentation in the middle of the data.
        poly_orders : list of int
            A list containing the polynomial orders used for the baseline fit.
            Must contain one more element than segment_borders. Default is
            [3, 3].
        y_at_borders : None, or list of float or None, optional
            May contain dependent variable values used as equality constraints
            at the segment borders. The fits of both touching segments are
            forced through the point given by the pair (segment border,
            y_at_border). The list entries may also be None to state that at a
            certain segment border, no constraint is to be applied. The default
            is [0, 0].
    """
    # Optionallly, spectrum data is smoothed before beaseline calculation. This
    # makes sense especially for baseline generation methods that have problems
    # with noise. Currently Savitzky-Golay only.
    if smoothing:
        savgol_window = kwargs.get('savgol_window', 9)
        savgol_order = kwargs.get('savol_order', 2)
        raw_data = smooth_spectra(raw_data, 'sav_gol',
                                  savgol_points=savgol_window,
                                  poly_order=savgol_order)

    # Transformation makes sense for spectra that cover a broad range of peak
    # intensities. Otherwise, small peaks may be more or less ignored during
    # baseline calculation. Currently LLS transformation only.
    if transform:
        spectra_minimum_value = raw_data.min()
        raw_data = transform_spectra(raw_data, 'log_log_sqrt')

    # wavenumbers are used for convex_hull, ModPoly, IModPoly, PPF, iALSS
    if 'wavenumbers' in kwargs:
        wavenumbers = kwargs.get('wavenumbers')
        ascending_wn = (wavenumbers[1]-wavenumbers[0]) > 0
    else:
        wavenumbers = np.arange(raw_data.shape[1])
        ascending_wn = True

    baseline_data = np.zeros_like(raw_data)
    baseline_modes = ['convex_hull', 'ALSS', 'iALSS', 'drPLS', 'SNIP',
                      'ModPoly', 'IModPoly', 'PPF']

    if mode == baseline_modes[0]:  # convex_hull
        # based on (but improved a bit)
        # https://dsp.stackexchange.com/questions/2725/
        # how-to-perform-a-rubberband-correction-on-spectroscopic-data

        if ascending_wn:
            raw_data = np.flip(raw_data, axis=0)
            wavenumbers = np.flip(wavenumbers)

        for ii, current_spectrum in enumerate(tqdm(raw_data)):
            hull_vertices = ConvexHull(
                np.array(list(zip(wavenumbers, current_spectrum)))).vertices

            # Rotate convex hull vertices until they start from the lowest one
            hull_vertices = np.roll(hull_vertices, -np.argmin(hull_vertices))

            # split vertices into upper and lower part
            hull_vertices_section_1 = hull_vertices[:np.argmax(hull_vertices)
                                                    + 1]
            hull_vertices_section_2 = np.sort(
                np.insert(hull_vertices[np.argmax(hull_vertices):], 0,
                          hull_vertices[0]))

            # calculate spectrum mean intensities of upper and lower vertices
            raw_mean_1 = np.mean(current_spectrum[hull_vertices_section_1])
            raw_mean_2 = np.mean(current_spectrum[hull_vertices_section_2])

            # Select lower vertices as baseline vertices
            if raw_mean_1 > raw_mean_2:
                baseline_vertices = hull_vertices_section_2
            else:
                baseline_vertices = hull_vertices_section_1

            # Create baseline using linear interpolation between vertices
            baseline_data[ii, :] = np.interp(
                wavenumbers, np.flip(wavenumbers[baseline_vertices]),
                np.flip(current_spectrum[baseline_vertices]))

        if ascending_wn:
            baseline_data = np.flip(baseline_data, axis=0)

    elif mode == baseline_modes[1]:  # ALSS
        # according to
        # "Baseline Correction with Asymmetric Least Squares Smoothing"
        # by P. Eilers and H. Boelens.
        # https://zanran_storage.s3.amazonaws.com/www.science.uva.nl/
        # ContentPages/443199618.pdf

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
                    w = p * (current_spectrum > z) + (1-p) * (
                        current_spectrum < z)
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
                    w = p * (current_spectrum > z) + (1-p) * (
                        current_spectrum < z)
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
                    w = 0.5 * (1 - np.exp(jj) * (d - (
                        -mean_negative + 2*sigma_negative))/sigma_negative / (
                            1 + np.abs(np.exp(jj) * (d - (
                                - mean_negative + 2 * sigma_negative)) /
                                sigma_negative)))
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

    elif mode in baseline_modes[5:8]:  # ModPoly, IModPoly, PPF
        # according to Applied Spectroscopy, 2007, 61 (11), 1225-1232.
        # without dev: Chemometrics and Intelligent Laboratory Systems 82
        #              (2006) 59– 65.
        #              Maybe also ModPoly from first source?

        # set mode specific parameters
        n_iter = kwargs.get('n_iter', 100)
        if mode in baseline_modes[5:7]:  # ModPoly, IModPoly
            poly_order = kwargs.get('poly_order', 5)
            fixed_points = kwargs.get('fixed_points', None)
            fixed_slopes = kwargs.get('fixed_slopes', None)
        if mode == baseline_modes[7]:  # PPF
            segment_borders = kwargs.get(
                'segment_borders', wavenumbers[len(wavenumbers)//2])
            poly_orders = kwargs.get('poly_orders', [3, 3])
            y_at_borders = kwargs.get('y_at_borders', [0, 0])
        #############################

        wavenumbers_start = wavenumbers
        # previous_dev = 0

        for ii, current_spectrum in enumerate(tqdm(raw_data)):
            wavenumbers = wavenumbers_start
            for jj in range(int(n_iter)):
                if mode in baseline_modes[5:7]:  # ModPoly, IModPoly
                    # The polynomial_fit method from pyRegression is only used
                    # if constraints are to be considered because the numpy
                    # polyfit method is faster.
                    if (fixed_points is not None) or (
                            fixed_slopes is not None):
                        fit_data, fit_coeffs = polynomial_fit(
                            wavenumbers, current_spectrum, poly_order,
                            fixed_points=fixed_points,
                            fixed_slopes=fixed_slopes)
                    else:
                        fit_coeffs = np.polynomial.polynomial.polyfit(
                            wavenumbers, current_spectrum, poly_order)
                        fit_data = np.polynomial.polynomial.polyval(
                            wavenumbers, fit_coeffs)
                else:  # PPF
                    fit_data, fit_coeffs = piecewise_polynomial_fit(
                        wavenumbers, current_spectrum, segment_borders,
                        poly_orders, y_at_borders=y_at_borders,
                        slope_at_borders=None)

                if mode == baseline_modes[5]:  # ModPoly
                    dev = 0
                else:  # IModPoly, PPF
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

            if mode in baseline_modes[5:7]:  # ModPoly, IModPoly
                baseline_data[ii, :] = np.polynomial.polynomial.polyval(
                    wavenumbers_start, fit_coeffs)
            else:  # PPF
                baseline_data[ii, :] = piecewise_polynomial(
                    wavenumbers_start, fit_coeffs,
                    segment_borders=segment_borders)

    # elif mode == baseline_modes[7]:  # PPF
    #     # according to Photonic Sensors 2018, 8(4), 332-340.

    #     # set mode specific parameters
    #     n_iter = kwargs.get('n_iter', 100)
    #     poly_order = kwargs.get('poly_order', 5)
    #     slope_threshold = kwargs.get('slope_threshold', 0.01)
    #     check_point_number = kwargs.get('check_point_number', 20)
    #     step_threshold = kwargs.get('step_threshold', 0.005)
    #     #############################

    #     if ascending_wn is False:
    #         raw_data = np.flip(raw_data, axis=0)
    #         wavenumbers = np.flip(wavenumbers)

    #     # derivative is calculated for later use as criterium to distinguish
    #     # between baseline and peaks
    #     raw_data_derivative = derivative(wavenumbers, raw_data)

    #     # The sign of the derivative changes from 1 to -1 at a peak maximum and
    #     # from -1 to 1 upon the next increase of the intensities after a peak
    #     derivative_sign_changes = np.diff(
    #             np.sign(raw_data_derivative), axis=1,
    #             append=raw_data_derivative[:, -1, np.newaxis])

    #     # Interation through each spectrum at a time because the spectra will
    #     # contain different amounts of segmentation points
    #     for ii, (current_spectrum, curr_deriv,
    #              current_sign_change) in enumerate(zip(
    #                  raw_data, raw_data_derivative, derivative_sign_changes)):

    #         # Indexes of first points after peaks where the derivative becomes
    #         # positive, those point to possible segmentation points
    #         peak_boundaries = np.where(current_sign_change == 2)[0]

    #         # Row and column index to select the next check_point_number points
    #         # to find out whether the slopes next to the potential segmentation
    #         # points are above the value of slope_threshold
    #         rows = np.arange(check_point_number)[np.newaxis]
    #         columns = peak_boundaries[:, np.newaxis] + np.arange(
    #             check_point_number) + 1

    #         # In case the check point indices extend over the end of the
    #         # spectrum, those indexes are set to the last one of the spectrum
    #         columns[columns>len(current_spectrum)-1] = len(current_spectrum)-1

    #         # Check points, the neighboring check_point_number points right of
    #         # peak_boundaries, are read out from the spectrum derivative
    #         check_points = np.tile(curr_deriv, check_point_number).reshape(
    #             check_point_number, -1)[rows, columns]

    #         # Absolute differences of slopes at the potential segmentation
    #         # points and the corresponding check points are calculated
    #         deriv_diffs = np.abs(
    #             check_points - curr_deriv[peak_boundaries][:, np.newaxis])

    #         # Only those segmetation point candidates are kept where none of
    #         # the next check_point_number points has a derivative greater than
    #         # slope_threshold
    #         segmentation_points = peak_boundaries[
    #             ~np.any(deriv_diffs > slope_threshold, axis=1)] + check_point_number

    #         # Segmentation points that extend over the spectrum end are set to
    #         # the spectrum end
    #         segmentation_points = np.where(
    #             segmentation_points>len(current_spectrum)-1,
    #             len(current_spectrum)-1, segmentation_points)

    #         # Segemtation point indexes at 0 and len(current_spectrum) are
    #         # added at the beginning and the end of segmentation_points
    #         segmentation_points = np.insert(segmentation_points, 0, 0)
    #         if len(segmentation_points) == 1:
    #             segmentation_points = np.insert(segmentation_points, 1,
    #                                             len(current_spectrum))
    #         else:
    #             segmentation_points[-1] = len(current_spectrum)
            
    #         # Drop duplicates that might have occurred due to far right
    #         # peak_boundaries
    #         segmentation_points = np.unique(segmentation_points)

    #         # Used as the condition in the while loop trying to reduce the
    #         # vertical steps/discontinuities between the different polynomial
    #         # functions
    #         step_above_thresh = True

    #         # This controls the extension of the fitted regions over the
    #         # segmentation points. Initialized with zeros, so the first fits
    #         # are defined by the segmentation points.
    #         segment_addition = np.zeros_like(segmentation_points)

    #         # The maximum extension of the fitted segments is given by the
    #         # distance of the segmentation points to the ends of the spectrum
    #         max_segment_addition = np.abs(
    #             np.array([0, len(current_spectrum)]) -
    #             segmentation_points[:, np.newaxis]).min(axis=1)

    #         while step_above_thresh:
    #             baseline_sections = []
    #             # The for loop iterates over the different fit regions. p1 and
    #             # p2 are the segmentation points of the current interval to be
    #             # fitted, a1 and a2 control if the fit runs over an extended
    #             # interval.
    #             for p1, p2, a1, a2 in zip(
    #                     segmentation_points[:-1], segmentation_points[1:],
    #                     segment_addition[:-1], segment_addition[1:]):
    #                 # print('a1, a2: ', a1, ',', a2)
    #                 # The indexes defining the subset of the spectrum to be
    #                 # fitted
    #                 fit_lim_1 = p1 - a1
    #                 fit_lim_2 = p2 + a2

    #                 # The fit itself making use of the IModPoly algorithm
    #                 curr_section = np.squeeze(generate_baseline(
    #                     current_spectrum[fit_lim_1:fit_lim_2][np.newaxis],
    #                     'IModPoly', smoothing=False, transform=False,
    #                     wavenumbers=wavenumbers[fit_lim_1:fit_lim_2],
    #                     poly_order=poly_order, n_iter=n_iter))

    #                 # The current fit result of the current spectrum subset is
    #                 # collected
    #                 if a2 == 0:
    #                     baseline_sections.append(curr_section[a1:])
    #                 else:
    #                     baseline_sections.append(curr_section[a1:-a2])

    #             # The different baseline sections are combined
    #             baseline_data[ii] = np.concatenate(baseline_sections)

    #             # The vertical steps/discontinuities between the individual
    #             # baseline sections are calculated
    #             vertical_steps = np.abs(
    #                 baseline_data[ii, segmentation_points[1:-1]-1] -
    #                 baseline_data[ii, segmentation_points[1:-1]])
    #             vertical_steps = np.insert(vertical_steps, 0, 0)
    #             vertical_steps = np.append(vertical_steps, 0)

    #             # Check if the abortion criterium for the while loop can be
    #             # triggered in case the discontiuities are all below the value
    #             # of step_threshold
    #             if vertical_steps.max() <= step_threshold:
    #                 step_above_thresh = False
    #             # Check if the segements have reached the end of the spectrum
    #             # and thus cannot be extended any more
    #             elif np.any(
    #                     segment_addition[1:-1] == max_segment_addition[1:-1]):
    #                 raise Exception('Reached the spectrum end while extending '
    #                                 'the fitted region in order to improve '
    #                                 'discontinuities, so PPF not possible '
    #                                 'with this step_thresh.')
    #             # Else at the segmentation points at which the vertical step is
    #             # above the threshold, the subset used for the fit is extended
    #             else:
    #                 segment_addition = np.where(
    #                     vertical_steps > step_threshold,
    #                     segment_addition+1,
    #                     segment_addition)

    #         # print('vertical_steps', vertical_steps)
    #         # print('segmentation_points', segmentation_points)
    #         # print('segment_addition', segment_addition)

    #     if ascending_wn is False:
    #         baseline_data = np.flip(baseline_data, axis=0)
    #         peak_boundaries = np.flip(peak_boundaries)

    #     # return check_point_number

    else:
        raise ValueError('No valid baseline mode entered. Allowed modes are '
                         '{0}'.format(baseline_modes))

    if transform:
        baseline_data = transform_spectra(
            baseline_data, 'log_log_sqrt', direction='inverse',
            min_value=spectra_minimum_value)

    return np.around(baseline_data, decimals=6)
