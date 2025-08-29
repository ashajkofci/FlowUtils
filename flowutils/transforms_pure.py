"""
Pure Python implementation of logicle and hyperlog transforms
Compatible with numpy v1.22
"""

import numpy as np


def _solve(b, w):
    """
    Solve equation: 2 * (ln(d) - ln(b)) + w * (b + d) = 0 for d
    This is used in the logicle transform parameter calculation.
    """
    if w == 0:
        return b
    
    # precision is the same as that of b
    tolerance = 2 * b * np.finfo(float).eps
    
    # bracket the root using bisection method
    d_lo = 0
    d_hi = b
    
    # bisection first step
    d = (d_lo + d_hi) / 2
    last_delta = d_hi - d_lo
    
    # evaluate f(w,b) = 2 * (ln(d) - ln(b)) + w * (b + d)
    f_b = -2 * np.log(b) + w * b
    f = 2 * np.log(d) + w * d + f_b
    last_f = np.nan
    
    for i in range(1, 40):
        # compute the derivative
        df = 2 / d + w
        
        # if Newton's method would step outside the bracket
        # or if it isn't converging quickly enough
        if (((d - d_hi) * df - f) * ((d - d_lo) * df - f) >= 0 or
            abs(1.9 * f) > abs(last_delta * df)):
            # take a bisection step
            delta = (d_hi - d_lo) / 2
            d = d_lo + delta
            if d == d_lo:
                return d
        else:
            # otherwise take a Newton's method step
            delta = f / df
            t = d
            d -= delta
            if d == t:
                return d
        
        # if we've reached the desired precision we're done
        if abs(delta) < tolerance:
            return d
        last_delta = delta

        # recompute the function
        f = 2 * np.log(d) + w * d + f_b
        if f == 0 or f == last_f:
            return d
        last_f = f

        # update the bracketing interval
        if f < 0:
            d_lo = d
        else:
            d_hi = d
    
    return -1


def _taylor_series(taylor_coeffs, x1, scale):
    """
    Compute Taylor series expansion around x1
    """
    x = scale - x1
    TAYLOR_LENGTH = len(taylor_coeffs)
    sum_val = taylor_coeffs[TAYLOR_LENGTH - 1] * x
    for i in range(TAYLOR_LENGTH - 2, -1, -1):
        sum_val = (sum_val + taylor_coeffs[i]) * x
    return sum_val


def _series_biexponential(params, scale):
    """
    Compute bi-exponential series for values near zero
    """
    return _taylor_series(params['taylor'], params['x1'], scale)


def _logicle_scale_single(value, t, w, m, a):
    """
    Apply logicle scaling to a single value
    """
    # Setup parameters
    T, W, M, A = t, w, m, a
    
    # actual parameters from bi-exponential paper
    w_param = W / (M + A)
    x2 = A / (M + A)
    x1 = x2 + w_param
    x0 = x2 + 2 * w_param
    b = (M + A) * np.log(10.)
    d = _solve(b, w_param)
    
    c_a = np.exp(x0 * (b + d))
    mf_a = np.exp(b * x1) - c_a / np.exp(d * x1)
    
    a_param = T / ((np.exp(b) - mf_a) - c_a / np.exp(d))
    c_param = c_a * a_param
    f = -mf_a * a_param
    
    # use Taylor series near x1 to avoid round off problems
    xTaylor = x1 + w_param / 4
    
    # compute coefficients of the Taylor series (16 is enough for full precision)
    TAYLOR_LENGTH = 16
    pos_coef = a_param * np.exp(b * x1)
    neg_coef = -c_param / np.exp(d * x1)
    
    taylor_coeffs = []
    for i in range(TAYLOR_LENGTH):
        pos_coef *= b / (i + 1)
        neg_coef *= -d / (i + 1)
        taylor_coeffs.append(pos_coef + neg_coef)
    
    taylor_coeffs[1] = 0  # exact result of Logicle condition
    
    params = {
        'T': T, 'W': W, 'M': M, 'A': A,
        'a': a_param, 'b': b, 'c': c_param, 'd': d, 'f': f,
        'w': w_param, 'x0': x0, 'x1': x1, 'x2': x2,
        'xTaylor': xTaylor, 'taylor': taylor_coeffs
    }
    
    return _scale(params, value)


def _scale(params, value):
    """
    Scale function for logicle transform
    """
    # handle true zero separately
    if value == 0:
        return params['x1']
    
    # reflect negative values
    negative = value < 0
    if negative:
        value = -value
    
    # initial guess at solution
    if value < params['f'] + params['a'] * np.exp(params['b'] * params['x1']):
        # near x1, use linear approximation
        x = params['x1'] + value * params['w'] / (params['f'] + params['a'] * np.exp(params['b'] * params['x1']))
    else:
        # otherwise use ordinary logarithm
        x = np.log(value / params['a']) / params['b']
    
    # try for double precision unless in extended range
    tolerance = 3 * np.finfo(float).eps
    if x > 1:
        tolerance = 3 * x * np.finfo(float).eps
    
    for i in range(40):
        # compute the function and its derivatives
        ae2bx = params['a'] * np.exp(params['b'] * x)
        ce2mdx = params['c'] / np.exp(params['d'] * x)
        
        if x < params['xTaylor']:
            # near zero use the Taylor series
            y = _series_biexponential(params, x) - value
        else:
            # this formulation has better round-off behavior
            y = (ae2bx + params['f']) - (ce2mdx + value)
        
        abe2bx = params['b'] * ae2bx
        cde2mdx = params['d'] * ce2mdx
        dy = abe2bx + cde2mdx
        ddy = params['b'] * abe2bx - params['d'] * cde2mdx
        
        # this is Halley's method with cubic convergence
        delta = y / (dy * (1 - y * ddy / (2 * dy * dy)))
        x -= delta
        
        # if we've reached the desired precision we're done
        if abs(delta) < tolerance:
            # handle negative arguments
            if negative:
                return 2 * params['x1'] - x
            else:
                return x
    
    # if we get here, scale did not converge
    return -1


def _logicle_inverse_single(value, t, w, m, a):
    """
    Apply inverse logicle scaling to a single value
    """
    # Setup parameters (same as forward transform)
    T, W, M, A = t, w, m, a
    
    w_param = W / (M + A)
    x2 = A / (M + A)
    x1 = x2 + w_param
    x0 = x2 + 2 * w_param
    b = (M + A) * np.log(10.)
    d = _solve(b, w_param)
    
    c_a = np.exp(x0 * (b + d))
    mf_a = np.exp(b * x1) - c_a / np.exp(d * x1)
    
    a_param = T / ((np.exp(b) - mf_a) - c_a / np.exp(d))
    c_param = c_a * a_param
    f = -mf_a * a_param
    
    xTaylor = x1 + w_param / 4
    
    # compute coefficients of the Taylor series
    TAYLOR_LENGTH = 16
    pos_coef = a_param * np.exp(b * x1)
    neg_coef = -c_param / np.exp(d * x1)
    
    taylor_coeffs = []
    for i in range(TAYLOR_LENGTH):
        pos_coef *= b / (i + 1)
        neg_coef *= -d / (i + 1)
        taylor_coeffs.append(pos_coef + neg_coef)
    
    taylor_coeffs[1] = 0  # exact result of Logicle condition
    
    params = {
        'T': T, 'W': W, 'M': M, 'A': A,
        'a': a_param, 'b': b, 'c': c_param, 'd': d, 'f': f,
        'w': w_param, 'x0': x0, 'x1': x1, 'x2': x2,
        'xTaylor': xTaylor, 'taylor': taylor_coeffs
    }
    
    return _logicle_inverse_scale(params, value)


def _logicle_inverse_scale(params, value):
    """
    Inverse scale function for logicle transform
    """
    # reflect negative scale regions
    negative = value < params['x1']
    if negative:
        value = 2 * params['x1'] - value

    # compute the bi-exponential
    if value < params['xTaylor']:
        # near x1, i.e., data zero use the series expansion
        inverse = _series_biexponential(params, value)
    else:
        # this formulation has better round-off behavior
        inverse = (params['a'] * np.exp(params['b'] * value) + params['f']) - params['c'] / np.exp(params['d'] * value)

    # handle scale for negative values
    if negative:
        return -inverse
    else:
        return inverse


def _hyperlog_scale_single(value, t, w, m, a):
    """
    Apply hyperlog scaling to a single value
    """
    # Setup parameters
    T, W, M, A = t, w, m, a
    
    # actual parameters
    w_param = W / (M + A)
    x2 = A / (M + A)
    x1 = x2 + w_param
    x0 = x2 + 2 * w_param
    b = (M + A) * np.log(10.)
    e0 = np.exp(b * x0)

    c_a = e0 / w_param
    f_a = np.exp(b * x1) + c_a * x1
    a_param = T / (np.exp(b) + c_a - f_a)

    c_param = c_a * a_param
    f = f_a * a_param
    
    xTaylor = x1 + w_param / 4
    
    # compute coefficients of the Taylor series
    TAYLOR_LENGTH = 16
    coef = a_param * np.exp(b * x1)
    
    taylor_coeffs = []
    for i in range(TAYLOR_LENGTH):
        coef *= b / (i + 1)
        taylor_coeffs.append(coef)
    
    taylor_coeffs[0] += c_param
    
    # Store inverse value for hyperlog
    is_negative = x0 < x1
    tmp_x0 = 2 * x1 - x0 if is_negative else x0
    
    if tmp_x0 < xTaylor:
        inverse = _taylor_series(taylor_coeffs, x1, tmp_x0)
    else:
        inverse = a_param * np.exp(b * tmp_x0) + c_param * tmp_x0
    
    if is_negative:
        inverse = -inverse
    
    params = {
        'T': T, 'W': W, 'M': M, 'A': A,
        'a': a_param, 'b': b, 'c': c_param, 'f': f,
        'w': w_param, 'x0': x0, 'x1': x1, 'x2': x2,
        'xTaylor': xTaylor, 'taylor': taylor_coeffs, 'inverse': inverse
    }
    
    return _hyperscale(params, value)


def _hyperscale(params, value):
    """
    Scale function for hyperlog transform
    """
    # handle true zero separately
    if value == 0:
        return params['x1']

    # reflect negative values
    negative = value < 0
    if negative:
        value = -value

    # initial guess at solution
    if value < params['inverse']:
        x = params['x1'] + value * params['w'] / params['inverse']
    else:
        # otherwise use ordinary logarithm
        x = np.log(value / params['a']) / params['b']

    # try for double precision unless in extended range
    tolerance = 3 * np.finfo(float).eps

    for i in range(10):
        ae2bx = params['a'] * np.exp(params['b'] * x)
        
        if x < params['xTaylor']:
            # near zero use the Taylor series
            y = _taylor_series(params['taylor'], params['x1'], x) - value
        else:
            # this formulation has better round-off behavior
            y = (ae2bx + params['c'] * x) - (params['f'] + value)

        abe2bx = params['b'] * ae2bx
        dy = abe2bx + params['c']
        ddy = params['b'] * abe2bx

        # this is Halley's method with cubic convergence
        delta = y / (dy * (1 - y * ddy / (2 * dy * dy)))
        x -= delta

        # if we've reached the desired precision we're done
        if abs(delta) < tolerance:
            # handle negative arguments
            if negative:
                return 2 * params['x1'] - x
            else:
                return x

    # if we get here, scale did not converge
    return -1


def _hyperlog_inverse_single(value, t, w, m, a):
    """
    Apply inverse hyperlog scaling to a single value
    """
    # Setup parameters (same as forward transform)
    T, W, M, A = t, w, m, a
    
    w_param = W / (M + A)
    x2 = A / (M + A)
    x1 = x2 + w_param
    x0 = x2 + 2 * w_param
    b = (M + A) * np.log(10.)
    e0 = np.exp(b * x0)

    c_a = e0 / w_param
    f_a = np.exp(b * x1) + c_a * x1
    a_param = T / (np.exp(b) + c_a - f_a)

    c_param = c_a * a_param
    f = f_a * a_param
    
    xTaylor = x1 + w_param / 4
    
    # compute coefficients of the Taylor series
    TAYLOR_LENGTH = 16
    coef = a_param * np.exp(b * x1)
    
    taylor_coeffs = []
    for i in range(TAYLOR_LENGTH):
        coef *= b / (i + 1)
        taylor_coeffs.append(coef)
    
    taylor_coeffs[0] += c_param
    
    # Store inverse value 
    is_negative = x0 < x1
    tmp_x0 = 2 * x1 - x0 if is_negative else x0
    
    if tmp_x0 < xTaylor:
        inverse = _taylor_series(taylor_coeffs, x1, tmp_x0)
    else:
        inverse = a_param * np.exp(b * tmp_x0) + c_param * tmp_x0
    
    if is_negative:
        inverse = -inverse
    
    params = {
        'T': T, 'W': W, 'M': M, 'A': A,
        'a': a_param, 'b': b, 'c': c_param, 'f': f,
        'w': w_param, 'x0': x0, 'x1': x1, 'x2': x2,
        'xTaylor': xTaylor, 'taylor': taylor_coeffs, 'inverse': inverse
    }
    
    return _hyperscale_inverse(params, value)


def _hyperscale_inverse(params, value):
    """
    Inverse scale function for hyperlog transform
    """
    # reflect negative scale regions
    negative = value < params['x1']
    if negative:
        value = 2 * params['x1'] - value

    if value < params['xTaylor']:
        # near x1, use the series expansion
        inverse = _taylor_series(params['taylor'], params['x1'], value)
    else:
        # this formulation has better roundoff behavior
        inverse = (params['a'] * np.exp(params['b'] * value) + params['c'] * value) - params['f']

    # handle scale for negative values
    if negative:
        return -inverse
    else:
        return inverse


def _logicle(y, t=262144, m=4.5, w=0.5, a=0):
    """
    Pure Python implementation of logicle transform
    """
    y = np.array(y, dtype='double')
    y_flat = y.flatten()
    result = np.empty_like(y_flat)
    
    for i in range(len(y_flat)):
        result[i] = _logicle_scale_single(y_flat[i], t, w, m, a)
    
    return result.reshape(y.shape)


def _logicle_inverse(y, t=262144, m=4.5, w=0.5, a=0):
    """
    Pure Python implementation of logicle inverse transform
    """
    y = np.array(y, dtype='double')
    y_flat = y.flatten()
    result = np.empty_like(y_flat)
    
    for i in range(len(y_flat)):
        result[i] = _logicle_inverse_single(y_flat[i], t, w, m, a)
    
    return result.reshape(y.shape)


def _hyperlog(y, t=262144, m=4.5, w=0.5, a=0):
    """
    Pure Python implementation of hyperlog transform
    """
    y = np.array(y, dtype='double')
    y_flat = y.flatten()
    result = np.empty_like(y_flat)
    
    for i in range(len(y_flat)):
        result[i] = _hyperlog_scale_single(y_flat[i], t, w, m, a)
    
    return result.reshape(y.shape)


def _hyperlog_inverse(y, t=262144, m=4.5, w=0.5, a=0):
    """
    Pure Python implementation of hyperlog inverse transform
    """
    y = np.array(y, dtype='double')
    y_flat = y.flatten()
    result = np.empty_like(y_flat)
    
    for i in range(len(y_flat)):
        result[i] = _hyperlog_inverse_single(y_flat[i], t, w, m, a)
    
    return result.reshape(y.shape)


def logicle(
        data,
        channel_indices,
        t=262144,
        m=4.5,
        w=0.5,
        a=0
):
    """
    Logicle transformation, implemented as defined in the
    GatingML 2.0 specification:

    logicle(x, T, W, M, A) = root(B(y, T, W, M, A) − x)

    where B is a modified bi-exponential function defined as:

    B(y, T, W, M, A) = ae^(by) − ce^(−dy) − f

    The Logicle transformation was originally defined in the publication:

        Moore WA and Parks DR. Update for the logicle data scale including operational
        code implementations. Cytometry A., 2012:81A(4):273–277.

    :param data: NumPy array of FCS event data. If a 1-D array, channel_indices option is ignored
    :param channel_indices: channel indices to transform (other channels returned in place, untransformed).
        If None, then all events will be transformed.
    :param t: parameter for the top of the linear scale (e.g. 262144)
    :param m: parameter for the number of decades the true logarithmic scale
        approaches at the high end of the scale
    :param w: parameter for the approximate number of decades in the linear region
    :param a: parameter for the additional number of negative decades

    :return: NumPy array of transformed events
    """
    data_copy = data.copy()

    if len(data.shape) == 1:
        data_copy = _logicle(data_copy, t, m, w, a)
    else:
        # run logicle scale for each channel separately
        if channel_indices is None:
            channel_indices = range(data.shape[1])
        for i in channel_indices:
            tmp = _logicle(data_copy[:, i].T, t, m, w, a)
            data_copy.T[i] = tmp

    return data_copy


def logicle_inverse(
        data,
        channel_indices,
        t=262144,
        m=4.5,
        w=0.5,
        a=0
):
    """
    Inverse of the Logicle transformation (see `logicle()` documentation for more details)

    :param data: NumPy array of FCS event data. If a 1-D array, channel_indices option is ignored
    :param channel_indices: channel indices to transform (other channels returned in place, untransformed).
        If None, then all events will be transformed.
    :param t: parameter for the top of the linear scale (e.g. 262144)
    :param m: parameter for the number of decades the true logarithmic scale
        approaches at the high end of the scale
    :param w: parameter for the approximate number of decades in the linear region
    :param a: parameter for the additional number of negative decades

    :return: NumPy array of transformed events
    """
    data_copy = data.copy()

    if len(data.shape) == 1:
        data_copy = _logicle_inverse(data_copy, t, m, w, a)
    else:
        # run inverse logicle for each channel separately
        if channel_indices is None:
            channel_indices = range(data.shape[1])
        for i in channel_indices:
            tmp = _logicle_inverse(data_copy[:, i].T, t, m, w, a)
            data_copy.T[i] = tmp

    return data_copy


def hyperlog(
        data,
        channel_indices,
        t=262144,
        m=4.5,
        w=0.5,
        a=0,
):
    """
    Hyperlog transformation, implemented as defined in the
    GatingML 2.0 specification:

    hyperlog(x, T, W, M, A) = root(EH(y, T, W, M, A) − x)

    where EH is defined as:

    EH(y, T, W, M, A) = ae^(by) + cy − f

    The Hyperlog transformation was originally defined in the publication:

        Bagwell CB. Hyperlog-a flexible log-like transform for negative, zero, and
        positive valued data. Cytometry A., 2005:64(1):34–42.

    :param data: NumPy array of FCS event data. If a 1-D array, channel_indices option is ignored
    :param channel_indices: channel indices to transform (other channels returned in place, untransformed).
        If None, then all events will be transformed.
    :param t: parameter for the top of the linear scale (e.g. 262144)
    :param m: parameter for desired number of decades
    :param w: parameter for the approximate number of decades in the linear region
    :param a: parameter for the additional number of negative decades

    :return: NumPy array of transformed events
    """
    data_copy = data.copy()

    if len(data.shape) == 1:
        data_copy = _hyperlog(data_copy, t, m, w, a)
    else:
        # run hyperlog scale for each channel separately
        if channel_indices is None:
            channel_indices = range(data.shape[1])
        for i in channel_indices:
            tmp = _hyperlog(data_copy[:, i].T, t, m, w, a)
            data_copy.T[i] = tmp

    return data_copy


def hyperlog_inverse(
        data,
        channel_indices,
        t=262144,
        m=4.5,
        w=0.5,
        a=0,
):
    """
    Inverse of the Hyperlog transformation, implemented as defined in the
    GatingML 2.0 specification (see hyperlog() documentation for more details).

    :param data: NumPy array of FCS event data. If a 1-D array, channel_indices option is ignored
    :param channel_indices: channel indices to transform (other channels returned in place, untransformed).
        If None, then all events will be transformed.
    :param t: parameter for the top of the linear scale (e.g. 262144)
    :param m: parameter for desired number of decades
    :param w: parameter for the approximate number of decades in the linear region
    :param a: parameter for the additional number of negative decades

    :return: NumPy array of transformed events
    """
    data_copy = data.copy()

    if len(data.shape) == 1:
        data_copy = _hyperlog_inverse(data_copy, t, m, w, a)
    else:
        # run hyperlog inverse for each channel separately
        if channel_indices is None:
            channel_indices = range(data.shape[1])
        for i in channel_indices:
            tmp = _hyperlog_inverse(data_copy[:, i].T, t, m, w, a)
            data_copy.T[i] = tmp
    return data_copy