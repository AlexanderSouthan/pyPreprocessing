# -*- coding: utf-8 -*-
"""
Provides functions for data transformation (currently only LLS) and
normalization.
"""

import numpy as np

from little_helpers.array_tools import closest_index


def transform(raw_data, mode, direction='direct', **kwargs):
    """
    Apply mathematical transformations to data.

    Parameters
    ----------
    raw_data : ndarray
        2D numpy array with the shape (N, M) containing N data rows to be
        smoothed. Each data row is represented by row in numpy array and
        contains M values. If only one data row is present, raw_data has the
        shape (1, M).
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


def normalize(raw_data, mode, factor=1, **kwargs):
    '''
    Normalize data such as spectra to a certain value.


    Parameters
    ----------
    raw_data : ndarray
        2D numpy array with the shape (N, M) containing N data rows to be
        normalized. Each data row is represented by row in numpy array and
        contains M values. If only one data row is present, raw_data has the
        shape (1, M).
    mode : string
        The mode of data normalization. Allowed modes are 'total_intensity'
        (total integral under the data is set to a specific value), 'integral'
        (integral under parts of the data is set to  a specific value), or
        'max_intensity' (data is divided by maximum intensity).
    factor : float, optional
        The value the normalized parameter has after the operation. The default
        is 1.
    **kwargs for the different modes
        mode is 'total_intensity' or 'integral':
            x_data : ndarray or list
                A 1D numpy array or list containing the x data (such as
                wavenumbers) corresponding to raw_data. Should be sorted in an
                ascending order.
        mode is 'integral':
            limits : list
                A list of two numbers giving the values in x_data which define
                the limits of the integration. If this is not given, the mode
                'integral' behaves identical to 'total_intensity'.

    Returns
    -------
    normalized_data : ndarray
        Normalized data with the same shape as raw_data.

    '''
    raw_data = np.asarray(raw_data)

    # list of allowed modes for normalization
    normalize_modes = ['total_intensity', 'integral', 'max_intensity']

    if mode in normalize_modes[0:2]:  # 'total_intensity', 'integral'
        if 'x_data' in kwargs:
            x_data = np.asarray(kwargs.get('x_data'))
        else:
            raise TypeError(
                'For mode \'total_intensity\' or \'integral\', x_data must be '
                'provided.')

        if 'limits' in kwargs:
            limits = kwargs.get('limits')
            limit_idx = closest_index(limits, x_data)
        else:
            limit_idx = [0, len(x_data)-1]

        integral = np.trapezoid(
            raw_data[:, limit_idx[0]:limit_idx[1]+1],
            x=x_data[limit_idx[0]:limit_idx[1]+1], axis=1)[:, np.newaxis]

        conversion_factor = 1/integral

    elif mode == normalize_modes[2]:  # 'max_intensity'
        conversion_factor = 1/raw_data.max(axis=1)[:, np.newaxis]

    else:
        raise ValueError('No valid normalization mode entered. Allowed modes '
                         'are {0}'.format(normalize_modes))

    normalized_data = raw_data * conversion_factor * factor
    return normalized_data
