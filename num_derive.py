#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 00:29:22 2020

@author: almami
"""

import numpy as np


def derivative(x_values, y_values, order=1):
    """
    Calculate the numerical derivative of data.

    Calculation is done by averaging the left and right derivative, only the
    two outermost data points are calculated with only a one-sided derivative.
    Therefore, the outermost order data points suffer from the numerical
    calculation and might be grossly incorrect.

    Parameters
    ----------
    x_values : ndarray
        The x values. Must be a 1D array of shape (N,).
    y_values : ndarray
        A 2D array containing the y data. Must be of shape (M, N) with M data
        rows to be derived that share the same x data.
    order : int, optional
        Gives the derivative order. Default is 1.

    Returns
    -------
    derivative : ndarray
        An ndarray of the shape (M, N) containing the derivative values.

    """
    x_spacing = np.diff(x_values)

    for ii in range(order):
        y_spacing = np.diff(y_values, axis=1)

        left_derivative = y_spacing/x_spacing
        right_derivative = np.roll(left_derivative, -1, axis=1)

        derivative = (left_derivative[:, :-1] + right_derivative[:, :-1])/2
        derivative = np.insert(derivative, 0, left_derivative[:, 0], axis=1)
        derivative = np.insert(derivative, derivative.shape[1],
                               left_derivative[:, -1], axis=1)
        y_values = derivative

    return derivative
