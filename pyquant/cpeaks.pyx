import numpy as np
import sys
from copy import deepcopy
cimport numpy as np
cimport cython
ctypedef np.float_t FLOAT_t
from scipy import optimize, integrate, stats
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import argrelmax, argrelmin, convolve, kaiser
from operator import itemgetter, attrgetter
from collections import OrderedDict
from pythomics.proteomics.config import NEUTRON

cdef int within_bounds(np.ndarray[FLOAT_t, ndim=1] res, np.ndarray[FLOAT_t, ndim=2] bnds):
    for i,j in zip(res, bnds):
        if j[0] is not None and i < j[0]:
            return 0
        if j[1] is not None and i > j[1]:
            return 0
    return 1

cdef np.ndarray[FLOAT_t, ndim=1] gauss(np.ndarray[FLOAT_t, ndim=1] x, float amp, float mu, float std):
    cdef np.ndarray[FLOAT_t, ndim=1] y = amp*np.exp(-(x - mu)**2/(2*std**2))
    return y

cpdef np.ndarray[FLOAT_t, ndim=1] gauss_ndim(np.ndarray[FLOAT_t, ndim=1] xdata, np.ndarray[FLOAT_t, ndim=1] params):
    cdef np.ndarray[FLOAT_t, ndim=1] amps
    cdef np.ndarray[FLOAT_t, ndim=1] mus
    cdef np.ndarray[FLOAT_t, ndim=1] sigmas
    amps, mus, sigmas = params[::3], params[1::3], params[2::3]
    cdef np.ndarray[FLOAT_t, ndim=1] data = np.zeros(len(xdata))
    for amp, mu, sigma in zip(amps, mus, sigmas):
        data += gauss(xdata, amp, mu, sigma)
    return data

cpdef float gauss_func(np.ndarray[FLOAT_t, ndim=1] guess, np.ndarray[FLOAT_t, ndim=1] xdata, np.ndarray[FLOAT_t, ndim=1] ydata):
    cdef np.ndarray[FLOAT_t, ndim=1] data = gauss_ndim(xdata, guess)
    # absolute deviation as our distance metric. Empirically found to give better results than
    # residual sum of squares for this data.
    cdef float residual = sum((ydata-data)**2)
    return residual

cpdef np.ndarray[FLOAT_t, ndim=1] bigauss_jac_old(np.ndarray[FLOAT_t, ndim=1] params, np.ndarray[FLOAT_t, ndim=1] x, np.ndarray[FLOAT_t, ndim=1] y):
    cdef np.ndarray[FLOAT_t, ndim=1] jac
    cdef np.ndarray[FLOAT_t, ndim=1] lx, ly, rx, ry
    cdef float amp, mu, stdl, stdr, sigma1, sigma2
    cdef int i
    jac = np.zeros_like(params)
    for i in xrange(params.shape[0]):
        if i%4 == 0:
            amp = params[i]
        elif i%4 == 1:
            mu = params[i]
        elif i%4 == 2:
            stdl = params[i]
        elif i%4 == 3:
            stdr = params[i]
            sigma1 = stdl
            sigma2 = stdr
            lx = x[x<=mu]
            ly = y[x<=mu]
            rx = x[x>mu]
            ry = y[x>mu]
            exp_term = np.exp(-((lx-mu)**2)/(2*sigma1**2))
            amp_exp_term = amp*exp_term
            prefix = 2*amp_exp_term
            # jac[i-3] += sum(prefix*(amp-ly*exp_term))
            jac[i-3] += sum(-2*exp_term*(ly-amp_exp_term))
            jac[i-2] += sum((-2*amp*(lx-mu)*exp_term*(ly-amp_exp_term))/(sigma1**2))
            jac[i-1] += sum((-2*amp*((lx-mu)**2)*exp_term*(ly-amp_exp_term))/(sigma1**3))
            exp_term = np.exp(-((rx-mu)**2)/(2*sigma2**2))
            amp_exp_term = amp*exp_term
            prefix = 2*amp_exp_term
            # There is NO right side contribution to the jacobian of the amplitude because rx is defined as
            # x>mu, therefore anything by the right side of the bigaussian function does not change the amplitude
            jac[i-3] += sum(-2*exp_term*(ry-amp_exp_term))
            jac[i-2] += sum((-2*amp*(rx-mu)*exp_term*(ry-amp_exp_term))/sigma2**2)
            jac[i] += sum((-2*amp*((rx-mu)**2)*exp_term*(ry-amp_exp_term))/(sigma2**3))
    return jac

cpdef np.ndarray[FLOAT_t, ndim=1] bigauss_jac(np.ndarray[FLOAT_t, ndim=1] params, np.ndarray[FLOAT_t, ndim=1] x, np.ndarray[FLOAT_t, ndim=1] y):
    cdef np.ndarray[FLOAT_t, ndim=1] jac, common, amp_term, left_common, right_common
    cdef np.ndarray[FLOAT_t, ndim=1] lx, ly, rx, ry
    cdef float amp, mu, sigma1, sigma2
    cdef int i
    jac = np.zeros_like(params)
    common = -bigauss_ndim(x, params) + y
    for i in xrange(params.shape[0]):
        if i%4 == 0:
            amp = params[i]
        elif i%4 == 1:
            mu = params[i]
        elif i%4 == 2:
            sigma1 = params[i]
        elif i%4 == 3:
            sigma2 = params[i]
            lx = x[x<=mu]
            amp_term = np.exp(-(-mu + lx)**2/(2*sigma1**2))
            left_common = common[x<=mu]
            jac[i-3] += sum(-2*left_common*amp_term)
            jac[i-2] += sum(amp*(2*mu - 2*lx)*left_common*amp_term/sigma1**2)
            jac[i-1] += sum(-2*amp*(-mu + lx)**2*left_common*amp_term/sigma1**3)
            rx = x[x>mu]
            amp_term = np.exp(-(-mu + rx)**2/(2*sigma2**2))
            right_common = common[x>mu]
            jac[i-3] += sum(-2*right_common*amp_term)
            jac[i-2] += sum(amp*(2*mu - 2*rx)*right_common*amp_term/sigma2**2)
            jac[i] += sum(-2*amp*(-mu + rx)**2*right_common*amp_term/sigma2**3)
    return jac

cpdef np.ndarray[FLOAT_t, ndim=1] gauss_jac_old(np.ndarray[FLOAT_t, ndim=1] params, np.ndarray[FLOAT_t, ndim=1] x, np.ndarray[FLOAT_t, ndim=1] y):
    cdef np.ndarray[FLOAT_t, ndim=1] jac
    cdef float amp, mu, sigma
    jac = np.zeros_like(params)
    for i in xrange(params.shape[0]):
        if i%3 == 0:
            amp = params[i]
        elif i%3 == 1:
            mu = params[i]
        elif i%3 == 2:
            sigma = params[i]
            exp_term = np.exp(-((x-mu)**2)/(2*sigma**2))
            amp_exp_term = amp*exp_term
            jac[i-2] += sum(-2*exp_term*(y-amp_exp_term))
            jac[i-1] += sum(-2*amp*(x-mu)*exp_term*(y-amp_exp_term)/(sigma**2))
            jac[i] += sum(-2*amp*((x-mu)**2)*exp_term*(y-amp_exp_term)/(sigma**3))
    return jac

cpdef np.ndarray[FLOAT_t, ndim=1] gauss_jac(np.ndarray[FLOAT_t, ndim=1] params, np.ndarray[FLOAT_t, ndim=1] x, np.ndarray[FLOAT_t, ndim=1] y):
    cdef np.ndarray[FLOAT_t, ndim=1] jac, common, amp_term
    cdef float amp, mu, sigma
    jac = np.zeros_like(params)
    common = -gauss_ndim(x, params) + y
    for i in xrange(params.shape[0]):
        if i%3 == 0:
            amp = params[i]
        elif i%3 == 1:
            mu = params[i]
        elif i%3 == 2:
            sigma = params[i]
            amp_term = np.exp(-(-mu + x)**2/(2*sigma**2))
            jac[i-2] += sum(-2*common*amp_term)
            jac[i-1] += sum(amp*(2*mu - 2*x)*common*amp_term/sigma**2)
            jac[i] += sum(-2*amp*(-mu + x)**2*common*amp_term/sigma**3)
    return jac

cdef np.ndarray[FLOAT_t, ndim=1] bigauss(np.ndarray[FLOAT_t, ndim=1] x, float amp, float mu, float stdl, float stdr):
    cdef float sigma1 = stdl
    cdef float sigma2 = stdr
    cdef np.ndarray[FLOAT_t, ndim=1] lx = x[x<=mu]
    cdef np.ndarray[FLOAT_t, ndim=1] left = amp*np.exp(-(lx-mu)**2/(2*sigma1**2))
    cdef np.ndarray[FLOAT_t, ndim=1] rx = x[x>mu]
    cdef np.ndarray[FLOAT_t, ndim=1] right = amp*np.exp(-(rx-mu)**2/(2*sigma2**2))
    cdef np.ndarray[FLOAT_t, ndim=1] y = np.concatenate([left, right], axis=0)
    return y

cpdef np.ndarray[FLOAT_t, ndim=1] bigauss_ndim(np.ndarray[FLOAT_t, ndim=1] xdata, np.ndarray[FLOAT_t, ndim=1] params):
    cdef np.ndarray[FLOAT_t, ndim=1] amps
    cdef np.ndarray[FLOAT_t, ndim=1] mus
    cdef np.ndarray[FLOAT_t, ndim=1] sigmasl
    cdef np.ndarray[FLOAT_t, ndim=1] sigmasr
    amps, mus, sigmasl, sigmasr = params[::4], params[1::4], params[2::4], params[3::4]
    cdef np.ndarray[FLOAT_t, ndim=1] data = np.zeros(len(xdata))
    for amp, mu, sigma1, sigma2 in zip(amps, mus, sigmasl, sigmasr):
        data += bigauss(xdata, amp, mu, sigma1, sigma2)
    return data

cpdef float bigauss_func(np.ndarray[FLOAT_t, ndim=1] guess, np.ndarray[FLOAT_t, ndim=1] xdata, np.ndarray[FLOAT_t, ndim=1] ydata):
    if any([np.isnan(i) for i in guess]):
        return np.inf
    cdef np.ndarray[FLOAT_t, ndim=1] data = bigauss_ndim(xdata, guess)
    cdef float residual = sum((ydata-data)**2)
    return residual

cpdef np.ndarray[FLOAT_t] fixedMeanFit(np.ndarray[FLOAT_t, ndim=1] xdata, np.ndarray[FLOAT_t, ndim=1] ydata,
                                       int peak_index=1, debug=False):
    cdef float rel_peak = ydata[peak_index]
    cdef float peak_loc = xdata[peak_index]
    cdef int peak_left, peak_right
    cdef float peak_min, peak_max, average, variance
    cdef np.ndarray[FLOAT_t, ndim=1] bnds

    ydata /= ydata.max()
    peak_left, peak_right = findPeak(convolve(ydata, kaiser(10, 14), mode='same'), peak_index)
    peak_min, peak_max = xdata[0], xdata[-1]
    # reset the fitting data to our bounds
    if peak_index == peak_right:
        peak_index -= peak_left-1
    else:
        peak_index -= peak_left
    xdata = xdata[peak_left:peak_right]
    ydata = ydata[peak_left:peak_right]
    if ydata.sum() == 0:
        return None
    min_spacing = min(np.diff(xdata))/2
    lb = np.fabs(peak_loc-xdata[0])
    rb = np.fabs(xdata[-1]-peak_loc)
    if lb < min_spacing:
        lb = min_spacing*5
    if rb < min_spacing:
        rb = min_spacing*5
    bnds = np.array([(rel_peak*0.75, 1.01) if rel_peak > 0 else (0.0, 1.0), (xdata[0], xdata[-1]), (min_spacing, lb), (min_spacing, rb)])
    average = np.average(xdata, weights=ydata)
    variance = np.sqrt(np.average((xdata-average)**2, weights=ydata))
    if variance == 0:
        # we have a singular peak if variance == 0, so set the variance to half of the x/y spacing
        if peak_index >= 1:
            variance = np.abs(peak_loc-xdata[peak_index-1])
        elif peak_index < len(xdata):
            variance = np.abs(xdata[peak_index+1]-peak_loc)
        else:
            # we have only 1 data point, most RT's fall into this width
            variance = 0.05
    if variance > xdata[peak_index]-peak_min or variance > peak_max-xdata[peak_index]:
        variance = xdata[peak_index]-peak_min
    cdef np.ndarray[FLOAT_t] guess = np.array([rel_peak, peak_loc, variance, variance])
    args = (xdata, ydata)
    base_opts = {'maxiter': 1000, 'ftol': 1e-20}
    routines = [('SLSQP', base_opts), ('TNC', base_opts), ('L-BFGS-B', base_opts)]
    routine, opts = routines.pop(0)
    try:
        results = [optimize.minimize(bigauss_func, guess, args, bounds=bnds, method=routine, options=opts, tol=1e-20)]
    except ValueError:
        print 'fitting error'
        import traceback
        print traceback.format_exc()
        print peak_loc
        print xdata.tolist()
        print ydata.tolist()
        print bnds
        results = []
    while routines:
        routine, opts = routines.pop(0)
        results.append(optimize.minimize(bigauss_func, guess, args, bounds=bnds, method=routine, options=opts, tol=1e-20))
    # cdef int n = len(xdata)
    cdef float lowest = -1
    cdef np.ndarray[FLOAT_t] best
    best = results[0].x

    for i in results:
        if within_bounds(i.x, bnds):
            if lowest == -1 or i.fun < lowest:
                best = i.x
    # cdef int k = len(best.x)
    # cdef float bic = n*np.log(best.fun/n)+k+np.log(n)
    # best.bic = bic
    return best

cpdef tuple fixedMeanFit2(np.ndarray[FLOAT_t, ndim=1] xdata, np.ndarray[FLOAT_t, ndim=1] ydata,
                                       int peak_index=1, debug=False):
    cdef float rel_peak, mval
    cdef float peak_loc = xdata[peak_index]
    cdef int peak_left, peak_right
    cdef np.ndarray[FLOAT_t, ndim=1] conv_y
    cdef float peak_min, peak_max, average, variance
    cdef np.ndarray[FLOAT_t, ndim=1] bnds

    mval = ydata.max()
    conv_y = gaussian_filter1d(convolve(ydata, kaiser(10, 14), mode='same'), 3, mode='constant')
    rel_peak = conv_y[peak_index]/conv_y.max()
    ydata /= mval
    peak_left, peak_right = 0, len(xdata)#findPeak(convolve(ydata, kaiser(10, 14), mode='same'), peak_index)
    peak_min, peak_max = xdata[0], xdata[-1]
    # reset the fitting data to our bounds
    if peak_index == peak_right:
        peak_index -= peak_left-1
    else:
        peak_index -= peak_left
    if debug:
        print('left is', peak_left, 'right is', peak_right)
        print('x', xdata.tolist(), 'becomes', xdata[peak_left:peak_right].tolist())
        print('y', ydata.tolist(), 'becomes', ydata[peak_left:peak_right].tolist())
    xdata = xdata[peak_left:peak_right]
    ydata = ydata[peak_left:peak_right]
    if ydata.sum() == 0:
        return None, None
    min_spacing = min(np.diff(xdata))/2
    lb = np.fabs(peak_loc-xdata[0])
    rb = np.fabs(xdata[-1]-peak_loc)
    if lb < min_spacing:
        lb = min_spacing*5
    if rb < min_spacing:
        rb = min_spacing*5
    bnds = np.array([(rel_peak*0.75, 1.01) if rel_peak > 0 else (0.0, 1.0), (xdata[0], xdata[-1]), (min_spacing, lb), (min_spacing, rb)])
    #print bnds, xdata, peak_loc
    average = np.average(xdata, weights=ydata)
    variance = np.sqrt(np.average((xdata-average)**2, weights=ydata))
    if variance == 0:
        # we have a singular peak if variance == 0, so set the variance to half of the x/y spacing
        if peak_index >= 1:
            variance = np.abs(peak_loc-xdata[peak_index-1])
        elif peak_index < len(xdata):
            variance = np.abs(xdata[peak_index+1]-peak_loc)
        else:
            # we have only 1 data point, most RT's fall into this width
            variance = 0.05
    if variance > xdata[peak_index]-peak_min or variance > peak_max-xdata[peak_index]:
        variance = xdata[peak_index]-peak_min
    if variance < min_spacing:
        variance = min_spacing
    cdef np.ndarray[FLOAT_t] guess = np.array([rel_peak, peak_loc, variance, variance])
    args = (xdata, ydata)
    base_opts = {'maxiter': 1000}
    routines = [('SLSQP', base_opts), ('TNC', base_opts), ('L-BFGS-B', base_opts)]
    routine, opts = routines.pop(0)

    if debug:
        print('guess and bounds', guess, bnds)
    try:
        results = [optimize.minimize(bigauss_func, guess, args, bounds=bnds, method=routine, options=opts, jac=bigauss_jac)]
    except ValueError:
        print 'fitting error'
        import traceback
        print traceback.format_exc()
        print peak_loc
        print xdata.tolist()
        print ydata.tolist()
        print bnds
        results = []
    while routines:# and results[-1].success == False:
        routine, opts = routines.pop(0)
        results.append(optimize.minimize(bigauss_func, guess, args, bounds=bnds, method=routine, options=opts, jac=bigauss_jac))
    # cdef int n = len(xdata)
    cdef float lowest = -1
    cdef np.ndarray[FLOAT_t] best
    if debug:
        print('fitting results', results)
    best_fit = results[0]
    for i in results:
        if within_bounds(i.x, bnds):
            if lowest == -1 or i.fun < lowest:
                best_fit = i
    best_fit.x[0]*=mval
    if debug:
        print('best fit', best_fit)
    # cdef int k = len(best.x)
    # cdef float bic = n*np.log(best.fun/n)+k+np.log(n)
    # best.bic = bic
    return best_fit.x, best_fit.fun


cdef tuple findPeak(np.ndarray[FLOAT_t, ndim=1] y, int srt):
    # check our SNR, if it's low, lessen our window
    cdef int left_offset = 1
    cdef int right_offset = 2
    cdef int lsrt = srt-left_offset if srt-left_offset > 0 else 0
    cdef int rsrt = srt+right_offset if srt+right_offset < len(y) else len(y)
    cdef float peak = y[srt]
    cdef int left = 0
    cdef np.ndarray[FLOAT_t] grad
    cdef int slope_shifts
    cdef float val
    cdef str ishift
    grad = y[lsrt:rsrt]
    try:
        shift = sum(np.sign(np.gradient(grad)))
    except ValueError:
        return 0, len(y)
    shift = 'left' if shift < 0 else 'right'
    ishift = shift
    slope_shifts = 0
    last_slope = -1
    for left in xrange(srt-1, -1, -1):
        val = y[left]
        grad = y[left-2:left+1]
        slope = None
        if len(grad) >= 2:
            slope = sum(np.sign(np.gradient(grad)))
            if slope < 0:
                slope = 'right'
            elif slope > 0:
                slope = 'left'
        if last_slope != -1:
            if last_slope != slope and slope != None:
                slope_shifts += 1
        last_slope = slope
        if ishift == 'right' and ishift == slope:
            break
        if ishift == 'left' and slope == 'right' and slope_shifts > 1:
            break
        if val == 0 or (val > peak and slope != 'right'):
            if val == 0 or shift != 'left':
                break
        elif shift == 'left' and slope != 'right':# slope != right logic: newsl
            shift = None

    cdef int right
    cdef float highest_val

    right = len(y)
    shift = ishift
    highest_val = peak
    slope_shifts = 0
    last_slope = -1
    for right in xrange(srt+1, len(y)):
        val = y[right]
        grad = y[right:right+3]
        slope = None
        if len(grad) >= 2:
            slope = sum(np.sign(np.gradient(grad)))
            if slope < 0:
                slope = 'right'
            elif slope > 0:
                slope = 'left'
        if last_slope != -1:
            if last_slope != slope and slope != None:
                slope_shifts += 1
        last_slope = slope
        if ishift == 'left' and ishift == slope:
            break
        if ishift == 'right' and slope == 'left' and slope_shifts > 1:
            break
        if val == 0 or (val > peak and slope != 'left'):
            if val > highest_val:
                highest_val = val
            if val == 0 or shift != 'right':
                if val == 0:
                    right += 1
                break
        elif shift == 'right' and slope != 'left':
            shift = None
            peak = highest_val
    return left, right

cpdef float get_ppm(float theoretical, float observed):
    return np.fabs(theoretical-observed)/theoretical


def find_nearest(np.ndarray[FLOAT_t, ndim=1] array, value):
    return array[find_nearest_index(array, value)]

def find_nearest_index(np.ndarray[FLOAT_t, ndim=1] array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx == 0:
        return 0
    elif idx == len(array):
        return -1
    elif np.fabs(value - array[idx-1]) < np.fabs(value - array[idx]):
        return idx-1
    else:
        return idx

def find_nearest_indices(np.ndarray[FLOAT_t, ndim=1] array, value):
    indices = np.searchsorted(array, value, side="left")
    out = []
    for search_index, idx in enumerate(indices):
        search_value = value[search_index]
        if idx == 0:
            out.append(0)
        elif idx == len(array):
            out.append(-1)
        elif np.fabs(search_value - array[idx-1]) < np.fabs(search_value - array[idx]):
            out.append(idx-1)
        else:
            out.append(idx)
    return out

cpdef np.ndarray[FLOAT_t, ndim=1] bigauss_hess(np.ndarray[FLOAT_t, ndim=1] params, np.ndarray[FLOAT_t, ndim=1] p, np.ndarray[FLOAT_t, ndim=1] x, np.ndarray[FLOAT_t, ndim=1] y):
    # https://www.wolframalpha.com/input/?i=second+derivative+of+((y-(a*exp(-(u-x)%5E2%2F(2s%5E2))))%5E2,+a)
    cdef np.ndarray[FLOAT_t, ndim=1] Hp
    cdef np.ndarray[FLOAT_t, ndim=1] lx, ly, rx, ry
    cdef object exp
    cdef float amp, mu, stdl, stdr, sigma1, sigma2
    cdef int i
    Hp = np.zeros_like(params)
    for i in xrange(params.shape[0]):
        if i%4 == 0:
            amp = params[i]
        elif i%4 == 1:
            mu = params[i]
        elif i%4 == 2:
            stdl = params[i]
        elif i%4 == 3:
            stdr = params[i]
            sigma1 = stdl/1.177
            sigma2 = stdr/1.177
            lx = x[x<=mu]
            ly = y[x<=mu]
            rx = x[x>mu]
            ry = y[x>mu]
            # jac[i-3] += sum(prefix*(amp-ly*exp_term))
            exp = np.exp
            Hp[i-3] += sum(2*exp(-(lx - mu)**2/sigma1**2))*p[i-3] + \
                       sum(2*exp(-(-mu + rx)**2/sigma2**2))*p[i-3] + \
                       sum(2*amp*(lx - mu)**2*exp(-(lx - mu)**2/sigma1**2)/sigma1**3 - 2*(lx - mu)**2*(-amp*exp(-(lx - mu)**2/(2*sigma1**2)) + ly)*exp(-(lx - mu)**2/(2*sigma1**2))/sigma1**3)*p[i-2] + \
                       sum(2*amp*(-mu + rx)**2*exp(-(-mu + rx)**2/sigma2**2)/sigma2**3 - 2*(-mu + rx)**2*(-amp*exp(-(-mu + rx)**2/(2*sigma2**2)) + ry)*exp(-(-mu + rx)**2/(2*sigma2**2))/sigma2**3)*p[i-2] + \
                       sum(-amp*(-2*lx + 2*mu)*exp(-(lx - mu)**2/sigma1**2)/sigma1**2 + (-2*lx + 2*mu)*(-amp*exp(-(lx - mu)**2/(2*sigma1**2)) + ly)*exp(-(lx - mu)**2/(2*sigma1**2))/sigma1**2)*p[i-1] + \
                       sum(-amp*(2*mu - 2*rx)*exp(-(-mu + rx)**2/sigma2**2)/sigma2**2 + (2*mu - 2*rx)*(-amp*exp(-(-mu + rx)**2/(2*sigma2**2)) + ry)*exp(-(-mu + rx)**2/(2*sigma2**2))/sigma2**2)*p[i]
            Hp[i-2] += sum(2*amp*(lx - mu)**2*exp(-(lx - mu)**2/sigma1**2)/sigma1**3 - 2*(lx - mu)**2*(-amp*exp(-(lx - mu)**2/(2*sigma1**2)) + ly)*exp(-(lx - mu)**2/(2*sigma1**2))/sigma1**3)*p[i-3] + \
                       sum(2*amp*(-mu + rx)**2*exp(-(-mu + rx)**2/sigma2**2)/sigma2**3 - 2*(-mu + rx)**2*(-amp*exp(-(-mu + rx)**2/(2*sigma2**2)) + ry)*exp(-(-mu + rx)**2/(2*sigma2**2))/sigma2**3)*p[i-3] + \
                       sum(2*amp**2*(lx - mu)**4*exp(-(lx - mu)**2/sigma1**2)/sigma1**6 + 6*amp*(lx - mu)**2*(-amp*exp(-(lx - mu)**2/(2*sigma1**2)) + ly)*exp(-(lx - mu)**2/(2*sigma1**2))/sigma1**4 - 2*amp*(lx - mu)**4*(-amp*exp(-(lx - mu)**2/(2*sigma1**2)) + ly)*exp(-(lx - mu)**2/(2*sigma1**2))/sigma1**6)*p[i-2] + \
                       sum(2*amp**2*(-mu + rx)**4*exp(-(-mu + rx)**2/sigma2**2)/sigma2**6 + 6*amp*(-mu + rx)**2*(-amp*exp(-(-mu + rx)**2/(2*sigma2**2)) + ry)*exp(-(-mu + rx)**2/(2*sigma2**2))/sigma2**4 - 2*amp*(-mu + rx)**4*(-amp*exp(-(-mu + rx)**2/(2*sigma2**2)) + ry)*exp(-(-mu + rx)**2/(2*sigma2**2))/sigma2**6)*p[i-2] + \
                       sum(-amp**2*(-2*lx + 2*mu)*(lx - mu)**2*exp(-(lx - mu)**2/sigma1**2)/sigma1**5 - 2*amp*(-2*lx + 2*mu)*(-amp*exp(-(lx - mu)**2/(2*sigma1**2)) + ly)*exp(-(lx - mu)**2/(2*sigma1**2))/sigma1**3 + amp*(-2*lx + 2*mu)*(lx - mu)**2*(-amp*exp(-(lx - mu)**2/(2*sigma1**2)) + ly)*exp(-(lx - mu)**2/(2*sigma1**2))/sigma1**5) * p[i-1] + \
                       sum(-amp**2*(-mu + rx)**2*(2*mu - 2*rx)*exp(-(-mu + rx)**2/sigma2**2)/sigma2**5 - 2*amp*(2*mu - 2*rx)*(-amp*exp(-(-mu + rx)**2/(2*sigma2**2)) + ry)*exp(-(-mu + rx)**2/(2*sigma2**2))/sigma2**3 + amp*(-mu + rx)**2*(2*mu - 2*rx)*(-amp*exp(-(-mu + rx)**2/(2*sigma2**2)) + ry)*exp(-(-mu + rx)**2/(2*sigma2**2))/sigma2**5)*p[i]
            Hp[i-1] += sum(-amp*(-2*lx + 2*mu)*exp(-(lx - mu)**2/sigma1**2)/sigma1**2 + (-2*lx + 2*mu)*(-amp*exp(-(lx - mu)**2/(2*sigma1**2)) + ly)*exp(-(lx - mu)**2/(2*sigma1**2))/sigma1**2)*p[i-3] + \
                       sum(-amp*(2*mu - 2*rx)*exp(-(-mu + rx)**2/sigma2**2)/sigma2**2 + (2*mu - 2*rx)*(-amp*exp(-(-mu + rx)**2/(2*sigma2**2)) + ry)*exp(-(-mu + rx)**2/(2*sigma2**2))/sigma2**2)*p[i-3] + \
                       sum(-amp**2*(-2*lx + 2*mu)*(lx - mu)**2*exp(-(lx - mu)**2/sigma1**2)/sigma1**5 - 2*amp*(-2*lx + 2*mu)*(-amp*exp(-(lx - mu)**2/(2*sigma1**2)) + ly)*exp(-(lx - mu)**2/(2*sigma1**2))/sigma1**3 + amp*(-2*lx + 2*mu)*(lx - mu)**2*(-amp*exp(-(lx - mu)**2/(2*sigma1**2)) + ly)*exp(-(lx - mu)**2/(2*sigma1**2))/sigma1**5)*p[i-2] + \
                       sum(-amp**2*(-mu + rx)**2*(2*mu - 2*rx)*exp(-(-mu + rx)**2/sigma2**2)/sigma2**5 - 2*amp*(2*mu - 2*rx)*(-amp*exp(-(-mu + rx)**2/(2*sigma2**2)) + ry)*exp(-(-mu + rx)**2/(2*sigma2**2))/sigma2**3 + amp*(-mu + rx)**2*(2*mu - 2*rx)*(-amp*exp(-(-mu + rx)**2/(2*sigma2**2)) + ry)*exp(-(-mu + rx)**2/(2*sigma2**2))/sigma2**5)*p[i-2] + \
                       sum(amp**2*(-2*lx + 2*mu)**2*exp(-(lx - mu)**2/sigma1**2)/(2*sigma1**4) + 2*amp*(-amp*exp(-(lx - mu)**2/(2*sigma1**2)) + ly)*exp(-(lx - mu)**2/(2*sigma1**2))/sigma1**2 - amp*(-2*lx + 2*mu)**2*(-amp*exp(-(lx - mu)**2/(2*sigma1**2)) + ly)*exp(-(lx - mu)**2/(2*sigma1**2))/(2*sigma1**4))*p[i-1]
            Hp[i] += sum(-amp*(-2*lx + 2*mu)*exp(-(lx - mu)**2/sigma1**2)/sigma1**2 + (-2*lx + 2*mu)*(-amp*exp(-(lx - mu)**2/(2*sigma1**2)) + ly)*exp(-(lx - mu)**2/(2*sigma1**2))/sigma1**2)*p[i-3] + \
                       sum(-amp*(2*mu - 2*rx)*exp(-(-mu + rx)**2/sigma2**2)/sigma2**2 + (2*mu - 2*rx)*(-amp*exp(-(-mu + rx)**2/(2*sigma2**2)) + ry)*exp(-(-mu + rx)**2/(2*sigma2**2))/sigma2**2)*p[i-3] + \
                       sum(-amp**2*(-2*lx + 2*mu)*(lx - mu)**2*exp(-(lx - mu)**2/sigma1**2)/sigma1**5 - 2*amp*(-2*lx + 2*mu)*(-amp*exp(-(lx - mu)**2/(2*sigma1**2)) + ly)*exp(-(lx - mu)**2/(2*sigma1**2))/sigma1**3 + amp*(-2*lx + 2*mu)*(lx - mu)**2*(-amp*exp(-(lx - mu)**2/(2*sigma1**2)) + ly)*exp(-(lx - mu)**2/(2*sigma1**2))/sigma1**5)*p[i-2] + \
                       sum(-amp**2*(-mu + rx)**2*(2*mu - 2*rx)*exp(-(-mu + rx)**2/sigma2**2)/sigma2**5 - 2*amp*(2*mu - 2*rx)*(-amp*exp(-(-mu + rx)**2/(2*sigma2**2)) + ry)*exp(-(-mu + rx)**2/(2*sigma2**2))/sigma2**3 + amp*(-mu + rx)**2*(2*mu - 2*rx)*(-amp*exp(-(-mu + rx)**2/(2*sigma2**2)) + ry)*exp(-(-mu + rx)**2/(2*sigma2**2))/sigma2**5)*p[i-2] + \
                       sum(amp**2*(2*mu - 2*rx)**2*exp(-(-mu + rx)**2/sigma2**2)/(2*sigma2**4) + 2*amp*(-amp*exp(-(-mu + rx)**2/(2*sigma2**2)) + ry)*exp(-(-mu + rx)**2/(2*sigma2**2))/sigma2**2 - amp*(2*mu - 2*rx)**2*(-amp*exp(-(-mu + rx)**2/(2*sigma2**2)) + ry)*exp(-(-mu + rx)**2/(2*sigma2**2))/(2*sigma2**4))*p[i]
    return Hp

cpdef np.ndarray[FLOAT_t, ndim=1] gauss_hess(np.ndarray[FLOAT_t, ndim=1] params, np.ndarray[FLOAT_t, ndim=1] p, np.ndarray[FLOAT_t, ndim=1] x, np.ndarray[FLOAT_t, ndim=1] y):
    cdef np.ndarray[FLOAT_t, ndim=1] Hp
    cdef float amp, mu, sigma
    Hp = np.zeros_like(params)
    for i in xrange(len(params)):
        if i%3 == 0:
            amp = params[i]
        elif i%3 == 1:
            mu = params[i]
        elif i%3 == 2:
            sigma = params[i]
            exp_term = np.exp(-((x-mu)**2)/(2*sigma**2))
            amp_exp_term = amp*exp_term
            Hp[i-2] += sum(2*exp_term)*p[i-2]+sum(-2*(2*amp - y*np.exp((mu - x)**2/(2*sigma**2)))*(mu - x)*np.exp(-(mu - x)**2/sigma**2)/sigma**2)*p[i-1]+sum(2*(2*amp - y*np.exp((mu - x)**2/(2*sigma**2)))*(mu - x)**2*np.exp(-(mu - x)**2/sigma**2)/sigma**3)*p[i]
            Hp[i-1] += sum(-2*(2*amp - y*np.exp((mu - x)**2/(2*sigma**2)))*(mu - x)*np.exp(-(mu - x)**2/sigma**2)/sigma**2)*p[i-2]+sum(2*amp*(sigma**2*(-amp + y*np.exp((mu - x)**2/(2*sigma**2))) + (2*amp - y*np.exp((mu - x)**2/(2*sigma**2)))*(mu - x)**2)*np.exp(-(mu - x)**2/sigma**2)/sigma**4)*p[i-1]+sum(2*amp*(mu - x)*(2*sigma**2*(amp - y*np.exp((mu - x)**2/(2*sigma**2))) - (2*amp - y*np.exp((mu - x)**2/(2*sigma**2)))*(mu - x)**2)*np.exp(-(mu - x)**2/sigma**2)/sigma**5)*p[i]
            Hp[i] += sum(2*(2*amp - y*np.exp((mu - x)**2/(2*sigma**2)))*(mu - x)**2*np.exp(-(mu - x)**2/sigma**2)/sigma**3)*p[i-2]+sum(2*amp*(mu - x)*(2*sigma**2*(amp - y*np.exp((mu - x)**2/(2*sigma**2))) - (2*amp - y*np.exp((mu - x)**2/(2*sigma**2)))*(mu - x)**2)*np.exp(-(mu - x)**2/sigma**2)/sigma**5)*p[i-1]+sum(2*amp*(mu - x)**2*(3*sigma**2*(-amp + y*np.exp((mu - x)**2/(2*sigma**2))) + (2*amp - y*np.exp((mu - x)**2/(2*sigma**2)))*(mu - x)**2)*np.exp(-(mu - x)**2/sigma**2)/sigma**6)*p[i]
    return Hp