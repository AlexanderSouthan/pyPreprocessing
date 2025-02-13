#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 19:59:07 2021

@author: Alexander Southan
"""

import numpy as np
import unittest

from src.pyPreprocessing import transform


class TestTransform(unittest.TestCase):

    def test_transform(self):
        x = np.linspace(0, 10, 1100)
        y = x**2 -30

        # test lls transformation
        y_lls = transform.transform([y], 'log_log_sqrt', direction='direct')
        y_lls_inv = transform.transform(
            y_lls, 'log_log_sqrt', direction='inverse', min_value=y.min())
        self.assertTrue(np.allclose(y, y_lls_inv[0]))

        # test errors
        self.assertRaises(
            ValueError, transform.transform, [y], 'log_log_sq',
            direction='direct')
        self.assertRaises(
            ValueError, transform.transform, [y], 'log_log_sq',
            direction='inverse')
        self.assertRaises(
            ValueError, transform.transform, [y], 'log_log_sqrt',
            direction='dir')

    def test_normalize(self):
        x = np.linspace(0, 10, 1100)
        y = x**2 -30

        y_norm = transform.normalize([y], 'total_intensity', x_data=x)
        self.assertAlmostEqual(np.trapezoid(y_norm, x=x, axis=1)[0], 1)

        y_norm_2 = transform.normalize([y], 'total_intensity', x_data=x,
                                       factor=3.25)
        self.assertAlmostEqual(np.trapezoid(y_norm_2, x=x, axis=1)[0], 3.25)

        # test errors
        self.assertRaises(ValueError, transform.normalize, [y], 'tot_int')
