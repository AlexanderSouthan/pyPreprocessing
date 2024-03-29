#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 19:59:07 2021

@author: Alexander Southan
"""

import numpy as np
import matplotlib.pyplot as plt
import unittest

from src.pyPreprocessing import baseline_correction


class TestBaselineCorrection(unittest.TestCase):

    def test_baseline_correction(self):

        # Calculate a simple spectrum
        centers = np.array([1200, 1600])
        amps = np.array([200, 100])
        widths = np.array([30, 500])
        noise_factor = 5

        wavenumbers = np.linspace(1000, 2000, 1001)
        intensities = (
            (amps*np.exp(-(wavenumbers[:, None]-centers)**2/widths)).sum(axis=1) +
            noise_factor * np.random.normal(size=wavenumbers.size))
        background = 0.0001* wavenumbers**2
        spectrum = intensities + background

        # Calculate baselines with different methods
        baseline_snip = baseline_correction.generate_baseline(
            spectrum[None], 'SNIP')
        baseline_convhull = baseline_correction.generate_baseline(
            spectrum[None], 'convex_hull', wavenumbers=wavenumbers)
        baseline_alss = baseline_correction.generate_baseline(
            spectrum[None], 'ALSS')
        baseline_ialss = baseline_correction.generate_baseline(
            spectrum[None], 'iALSS', wavenumbers=wavenumbers)
        baseline_drpls = baseline_correction.generate_baseline(
            spectrum[None], 'drPLS')
        baseline_modpoly = baseline_correction.generate_baseline(
            spectrum[None], 'ModPoly', wavenumbers=wavenumbers)
        baseline_imodpoly = baseline_correction.generate_baseline(
            spectrum[None], 'IModPoly', wavenumbers=wavenumbers)
        baseline_ppf = baseline_correction.generate_baseline(
            spectrum[None], 'PPF', wavenumbers=wavenumbers)

        spectrum_corrected = baseline_correction.correct_baseline(
            spectrum[None], 'SNIP')

        self.assertTrue(np.all(spectrum_corrected == spectrum-baseline_snip))

        # test with transformation
        baseline_transform = baseline_correction.generate_baseline(
            spectrum[None], 'SNIP', transform=True)

        # test with flipped spectrum
        baseline_snip_desc = baseline_correction.generate_baseline(
            spectrum[::-1][None], 'SNIP')
        self.assertTrue(np.all(baseline_snip_desc[0, ::-1] == baseline_snip))

        # test with descending wavenumbers
        baseline_modpoly_desc = baseline_correction.generate_baseline(
            spectrum[::-1][None], 'ModPoly', wavenumbers=wavenumbers[::-1])
        self.assertTrue(
            np.all(baseline_modpoly_desc[0, ::-1] == baseline_modpoly))

        # PPF with fixed border values
        baseline_ppf_fixed = baseline_correction.generate_baseline(
            spectrum[None], 'PPF', wavenumbers=wavenumbers, y_at_borders=[250],
            segment_borders=[1400.0])
        wn_idx = np.argmax(wavenumbers==1400)
        self.assertEqual(baseline_ppf_fixed[0, wn_idx], 250)

        # ModPoly with fixed points
        baseline_modpoly_fixed = baseline_correction.generate_baseline(
            spectrum[None], 'ModPoly', wavenumbers=wavenumbers,
            fixed_points=[[1400, 250]])
        self.assertEqual(baseline_modpoly_fixed[0, wn_idx], 250)

        # test error messages
        self.assertRaises(ValueError, baseline_correction.generate_baseline,
                          spectrum[None], 'ModPoy')

        plt.plot(wavenumbers, spectrum)
        plt.plot(wavenumbers, baseline_snip.T, label='SNIP')
        plt.plot(wavenumbers, baseline_convhull.T, label='convex hull')
        plt.plot(wavenumbers, baseline_alss.T, label='ALSS')
        plt.plot(wavenumbers, baseline_ialss.T, label='iALSS')
        plt.plot(wavenumbers, baseline_drpls.T, label='drPLS')
        plt.plot(wavenumbers, baseline_modpoly.T, label='ModPoly')
        plt.plot(wavenumbers, baseline_imodpoly.T, label='IModPoly')
        plt.plot(wavenumbers, baseline_ppf.T, label='PPF')
        plt.legend()
