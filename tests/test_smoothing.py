#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 19:59:07 2021

@author: Alexander Southan
"""

import numpy as np
import matplotlib.pyplot as plt
import unittest

from src.pyPreprocessing import smoothing


class TestSmoothing(unittest.TestCase):

    def test_smoothing(self):
        x = np.linspace(0, 10, 1100)
        noise = np.random.normal(size=(50, len(x)))

        x_interp, noise_savgol = smoothing.smoothing(
            noise, 'sav_gol', interpolate=True,
            x_coordinate=x, return_type='interp', savgol_points=10,
            window=15, data_points=1200, point_mirror=True)
        self.assertEqual(len(x_interp), len(noise_savgol.T))
        self.assertEqual(len(x_interp), 1200)
        self.assertTrue(noise.std() > noise_savgol.std())

        x_interp_2, noise_savgol_2 = smoothing.smoothing(
            noise, 'sav_gol', interpolate=True,
            x_coordinate=x, return_type='orig', savgol_points=10,
            window=15, data_points=1200, point_mirror=True)
        self.assertEqual(len(x_interp_2), len(noise_savgol_2.T))
        self.assertEqual(len(x_interp_2), 1100)
        self.assertTrue(noise.std() > noise_savgol_2.std())

        noise_rollingmedian = smoothing.smoothing(
            noise, 'rolling_median', window=10)
        self.assertTrue(noise.std() > noise_rollingmedian.std())

        noise_pca = smoothing.smoothing(noise, 'pca', pca_components=2)
        self.assertTrue(noise.std() > noise_pca.std())

        noise_weightedaverage, _ = smoothing.smoothing(
            noise, 'weighted_moving_average')
        self.assertTrue(noise.std() > noise_weightedaverage.std())

        # test with only one dataset
        noise_rollingmedian_single = smoothing.smoothing(
            noise[[0]], 'rolling_median', window=10)

        # test errors
        self.assertRaises(ValueError, smoothing.smoothing, noise[[0]],
                          'roling_median')
        self.assertRaises(ValueError, smoothing.smoothing, noise, 'sav_gol',
                          interpolate=True, x_coordinate=x, return_type='irp',
                          savgol_points=10, window=15, data_points=1100,
                          point_mirror=True)

    def test_filtering(self):
        x = np.linspace(0, 10, 1100)
        noise = np.random.normal(size=(50, len(x)))
        noise[:, 500] = 300
        noise[:, 700] = -300

        # test spike filter
        noise_spike = smoothing.filtering(noise, 'spike_filter')
        self.assertTrue(np.all(np.isnan(noise_spike[:, 500])))
        self.assertTrue(np.all(np.isnan(noise_spike[:, 700])))

        noise_spike_2 = smoothing.filtering(
            noise, 'spike_filter', fill='mov_avg',
            weights=[1, 0.2, 1, 0, 0.5, 1, 1])
        check_avg = (noise[:, 497] + 0.2 * noise[:, 498] + noise[:, 499] + 
                     0.5 * noise[:, 501] + noise[:, 502] + noise[:, 503])/4.7
        self.assertTrue(np.all(noise_spike_2[:, 500]==check_avg))

        noise_spike_3 = smoothing.filtering(
            noise, 'spike_filter', fill='zeros')
        self.assertTrue(np.all(noise_spike_3[:, 500] == 0))
        self.assertTrue(np.all(noise_spike_3[:, 700] == 0))

        # test maximum threshold
        noise_maxthresh = smoothing.filtering(noise, 'max_thresh',
                                              max_thresh=299)
        self.assertTrue(np.all(np.isnan(noise_maxthresh[:, 500])))
        self.assertFalse(np.any(np.isnan(noise_maxthresh[:, 700])))

        noise_maxthresh_2 = smoothing.filtering(noise, 'max_thresh',
                                              max_thresh=301)
        self.assertFalse(np.any(np.isnan(noise_maxthresh_2[:, 500])))
        self.assertFalse(np.any(np.isnan(noise_maxthresh_2[:, 700])))

        # test minimum threshold
        noise_minthresh = smoothing.filtering(noise, 'min_thresh',
                                              min_thresh=-299)
        self.assertFalse(np.any(np.isnan(noise_minthresh[:, 500])))
        self.assertTrue(np.all(np.isnan(noise_minthresh[:, 700])))

        noise_minthresh_2 = smoothing.filtering(noise, 'min_thresh',
                                              min_thresh=-301)
        self.assertFalse(np.any(np.isnan(noise_minthresh_2[:, 500])))
        self.assertFalse(np.any(np.isnan(noise_minthresh_2[:, 700])))

        # test errors
        self.assertRaises(ValueError, smoothing.filtering, noise,
                          'max_thresh', fill='mov_avg')
        self.assertRaises(ValueError, smoothing.filtering, noise,
                          'max_thresh', fill='zero')
        self.assertRaises(ValueError, smoothing.filtering, noise,
                          'spike_fil')