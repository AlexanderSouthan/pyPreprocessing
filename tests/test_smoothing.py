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
        y = x**2 + noise

        x_interp, noise_savgol = smoothing.smoothing(
            noise, 'sav_gol', interpolate=True,
            x_coordinate=x, return_type='interp', savgol_points=10,
            window=15, data_points=1100, point_mirror=True)

        noise_rollingmedian = smoothing.smoothing(
            noise, 'rolling_median', window=10)

        noise_pca = smoothing.smoothing(noise, 'pca', pca_components=2)

        noise_weightedaverage, _ = smoothing.smoothing(
            noise, 'weighted_moving_average')

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