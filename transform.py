# -*- coding: utf-8 -*-
"""
Provides functions for data transformation, currently only LLS.
"""

import numpy as np


def transform(raw_data, mode, direction='direct', **kwargs):
    """
    Applies mathematical transformations to data.

    Parameters
    ----------
    raw_data : ndarray
        2D numpy array with the shape (N,M) containing N data rows to be
        smoothed. Each data row is represented by row in numpy array and
        contains M values. If only one data row is present, raw_data has the
        shape (1,M).
    mode : str
        Maths used for transformation. Allowed mode is 'log_log_sqrt' only at
        the moment which first takes the square root and then does the
        logarithm twice.
    direction : str, optional
        Gives the direction of the tranformation. If 'direct', the data is
        transformed, if 'inverse', the inverse of the transformation is
        calculated. The default is 'direct'.
    **kwargs for the different modes
        mode is 'log_log_sqrt' and direction is 'inverse':
            min_value : float
                Original minimum value of the data before transformation. Has
                to be known because it is lost upon transformation. Default is
                1.

    Raises
    ------
    ValueError
        If the value passed as mode or direction is not understood.

    Returns
    -------
    raw_data : ndarray
        Transformed data with the same shape as raw_data.

    """
    # list of allowed modes for data transformation
    transform_modes = ['log_log_sqrt']

    if direction == 'direct':
        if mode == transform_modes[0]:
            minimum_value = np.min(raw_data)
            raw_data -= minimum_value
            raw_data = np.log(np.log(np.sqrt(raw_data + 1) + 1) + 1)
        else:
            raise ValueError('No valid transform mode entered. Allowed modes '
                             'are {0}'.format(transform_modes))

    elif direction == 'inverse':
        if mode == transform_modes[0]:
            minimum_value = kwargs.get('min_value', 1)
            raw_data = (np.exp(np.exp(raw_data) - 1) - 1)**2 - 1
            raw_data += minimum_value
        else:
            raise ValueError('No valid transform mode entered. Allowed modes '
                             'are {0}'.format(transform_modes))
    else:
        raise ValueError('No valid transform direction entered. Allowed '
                         'directions are [\'direct\', \'inverse\']')

    return raw_data
