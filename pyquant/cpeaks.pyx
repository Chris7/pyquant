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

cdef np.ndarray[FLOAT_t, ndim=1] adjust_baseline(np.ndarray[FLOAT_t, ndim=1] x, float slope, float intercept, float left, float right):
    cdef np.ndarray[FLOAT_t, ndim=1] baseline = slope*x+intercept
    # adjust y's baseline for area w/in the curve
    # baseline[x<=left] = 0
    # baseline[x>=right] = 0
    return baseline

cdef np.ndarray[FLOAT_t, ndim=1] gauss_bl(np.ndarray[FLOAT_t, ndim=1] x, float amp, float mu, float std, float slope, float intercept):
    cdef np.ndarray[FLOAT_t, ndim=1] y = amp*np.exp(-(x - mu)**2/(2*std**2))
    cdef np.ndarray[FLOAT_t, ndim=1] baseline = adjust_baseline(x, slope, intercept, mu-std*2, mu+std*2)
    y += baseline
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

cpdef np.ndarray[FLOAT_t, ndim=1] gauss_bl_ndim(np.ndarray[FLOAT_t, ndim=1] xdata, np.ndarray[FLOAT_t, ndim=1] params):
    cdef np.ndarray[FLOAT_t, ndim=1] amps
    cdef np.ndarray[FLOAT_t, ndim=1] mus
    cdef np.ndarray[FLOAT_t, ndim=1] sigmas
    cdef np.ndarray[FLOAT_t, ndim=1] slopes
    cdef np.ndarray[FLOAT_t, ndim=1] intercepts
    amps, mus, sigmas, slopes, intercepts = params[::5], params[1::5], params[2::5], params[3::5], params[4::5]
    cdef np.ndarray[FLOAT_t, ndim=1] data = np.zeros(len(xdata))
    for amp, mu, sigma, slope, intercept in zip(amps, mus, sigmas, slopes, intercepts):
        data += gauss_bl(xdata, amp, mu, sigma, slope, intercept)
    return data

cpdef float gauss_func(np.ndarray[FLOAT_t, ndim=1] guess, np.ndarray[FLOAT_t, ndim=1] xdata, np.ndarray[FLOAT_t, ndim=1] ydata):
    cdef np.ndarray[FLOAT_t, ndim=1] data = gauss_ndim(xdata, guess)
    # absolute deviation as our distance metric. Empirically found to give better results than
    # residual sum of squares for this data.
    cdef float residual = sum((ydata-data)**2)
    return residual

cpdef float gauss_bl_func(np.ndarray[FLOAT_t, ndim=1] guess, np.ndarray[FLOAT_t, ndim=1] xdata, np.ndarray[FLOAT_t, ndim=1] ydata):
    # absolute deviation as our distance metric. Empirically found to give better results than
    # residual sum of squares for this data.
    cdef float residual = 0
    cdef np.ndarray[FLOAT_t, ndim=1] amps
    cdef np.ndarray[FLOAT_t, ndim=1] mus
    cdef np.ndarray[FLOAT_t, ndim=1] sigmas
    cdef np.ndarray[FLOAT_t, ndim=1] slopes
    cdef np.ndarray[FLOAT_t, ndim=1] intercepts
    amps, mus, sigmas, slopes, intercepts = guess[::5], guess[1::5], guess[2::5], guess[3::5], guess[4::5]
    cdef np.ndarray[FLOAT_t, ndim=1] data = np.zeros(len(xdata))
    for amp, mu, sigma, slope, intercept in zip(amps, mus, sigmas, slopes, intercepts):
        data = gauss_bl(xdata, amp, mu, sigma, slope, intercept)
        mask = np.where((mu-sigma*3<=xdata) & (xdata<=mu+sigma*3))
        residual += sum((ydata[mask]-data[mask])**2)
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

cdef np.ndarray[FLOAT_t, ndim=1] bigauss_bl(np.ndarray[FLOAT_t, ndim=1] x, float amp, float mu, float stdl, float stdr, float slope, float intercept):
    cdef float sigma1 = stdl
    cdef float sigma2 = stdr
    cdef np.ndarray[FLOAT_t, ndim=1] lx = x[x<=mu]
    cdef np.ndarray[FLOAT_t, ndim=1] left = amp*np.exp(-(lx-mu)**2/(2*sigma1**2))
    cdef np.ndarray[FLOAT_t, ndim=1] rx = x[x>mu]
    cdef np.ndarray[FLOAT_t, ndim=1] right = amp*np.exp(-(rx-mu)**2/(2*sigma2**2))
    cdef np.ndarray[FLOAT_t, ndim=1] y = np.concatenate([left, right], axis=0)

    # adjust y's baseline for area w/in the curve
    cdef np.ndarray[FLOAT_t, ndim=1] baseline = adjust_baseline(x, slope, intercept, mu-sigma1*2, mu+sigma2*2)
    y += baseline
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

cpdef np.ndarray[FLOAT_t, ndim=1] bigauss_bl_ndim(np.ndarray[FLOAT_t, ndim=1] xdata, np.ndarray[FLOAT_t, ndim=1] params):
    cdef np.ndarray[FLOAT_t, ndim=1] amps
    cdef np.ndarray[FLOAT_t, ndim=1] mus
    cdef np.ndarray[FLOAT_t, ndim=1] sigmasl
    cdef np.ndarray[FLOAT_t, ndim=1] sigmasr
    cdef np.ndarray[FLOAT_t, ndim=1] bl_slopes
    cdef np.ndarray[FLOAT_t, ndim=1] bl_intercepts
    amps, mus, sigmasl, sigmasr, bl_slopes, bl_intercepts = params[::6], params[1::6], params[2::6], params[3::6], params[4::6], params[5::6]
    cdef np.ndarray[FLOAT_t, ndim=1] data = np.zeros(len(xdata))
    for amp, mu, sigma1, sigma2, bl_slope, bl_intercept in zip(amps, mus, sigmasl, sigmasr, bl_slopes, bl_intercepts):
        data += bigauss_bl(xdata, amp, mu, sigma1, sigma2, bl_slope, bl_intercept)
    return data

cpdef float bigauss_func(np.ndarray[FLOAT_t, ndim=1] guess, np.ndarray[FLOAT_t, ndim=1] xdata, np.ndarray[FLOAT_t, ndim=1] ydata):
    if any([np.isnan(i) for i in guess]):
        return np.inf
    cdef np.ndarray[FLOAT_t, ndim=1] data = bigauss_ndim(xdata, guess)
    cdef float residual = sum((ydata-data)**2)
    return residual

cpdef float bigauss_bl_func(np.ndarray[FLOAT_t, ndim=1] guess, np.ndarray[FLOAT_t, ndim=1] xdata, np.ndarray[FLOAT_t, ndim=1] ydata):
    if any([np.isnan(i) for i in guess]):
        return np.inf
    cdef np.ndarray[FLOAT_t, ndim=1] data = bigauss_bl_ndim(xdata, guess)
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

cpdef np.ndarray[FLOAT_t, ndim=1] gauss_hess(np.ndarray[FLOAT_t, ndim=1] params, np.ndarray[FLOAT_t, ndim=1] x, np.ndarray[FLOAT_t, ndim=1] y):
    cdef np.ndarray[FLOAT_t, ndim=1] common
    cdef np.ndarray[FLOAT_t, ndim=2] H
    H = np.zeros((params.shape[0], params.shape[0]))
    common = -1*gauss_ndim(x, params) + y
    for i in xrange(params.shape[0]):
        if i%3 == 0:
            amp = params[i]
        elif i%3 == 1:
            mu = params[i]
        elif i%3 == 2:
            sigma = params[i]
            for j in xrange(0, params.shape[0], 3):
                hess_amp, hess_mu, hess_sigma = params[j:j+3]
                # the diagonal in general refers to the hessian with partial
                # derivatives against itself (so a 3x3 or 4x4 area)
                diagonal = i == j+2
                # aa
                H[i-2, j] = sum(2 * np.exp((-(-mu+x)**2)/(sigma**2))) if diagonal else sum(2 * np.exp((-(-mu+x)**2)/(2*sigma**2)) * np.exp((-(-hess_mu+x)**2)/(2*hess_sigma**2)))
                # au
                H[i-2, j+1] = sum(-2*amp*(mu-x)/(sigma**2) * np.exp((-(-mu+x)**2)/(sigma**2)) + 2*(mu-x)/(sigma**2)*common*np.exp(-((-mu+x)**2)/(2*sigma**2))) if diagonal else sum(-2*hess_amp/(hess_sigma**2)*(hess_mu-x)*np.exp(-((-mu+x)**2)/(2*sigma**2))*np.exp(-((-hess_mu+x)**2)/(2*hess_sigma**2)))
                # as
                H[i-2, j+2] = sum(2*amp*(-mu+x)**2/(sigma**3)*np.exp(-(-mu+x)**2/(sigma**2)) - 2*((-mu+x)**2)/(sigma**3)*common*np.exp(-((-mu+x)**2)/(2*sigma**2))) if diagonal else sum(2*hess_amp*(-hess_mu+x)**2/(hess_sigma**3) * np.exp(-((-mu+x)**2)/(2*sigma**2)) * np.exp(-((-hess_mu+x)**2)/(2*hess_sigma**2)))
                # ua
                H[i-1, j] = sum(-2*amp*(mu-x)/(sigma**2)*np.exp(-((-mu+x)**2)/(sigma**2)) + 2*(mu-x)/(sigma**2)*common*np.exp(-((-mu+x)**2)/(2*sigma**2))) if diagonal else sum(-2*amp*(mu-x)/(sigma**2)*np.exp(-((-mu+x)**2)/(2*sigma**2)) * np.exp(-((-hess_mu+x)**2)/(2*hess_sigma**2)))#2*amp*(mu-x)/(2*sigma**2)*np.exp(-((-mu+x)**2)/(2*sigma**2)) * 2*hess_amp*(hess_mu-x)/(2*hess_sigma**2)*np.exp(-((-hess_mu+x)**2)/(2*hess_sigma**2)))
                # uu
                H[i-1, j+1] = sum(2*amp*(amp*(mu - x)**2*np.exp(-(mu - x)**2/sigma**2)/sigma**2 + common*np.exp(-(mu - x)**2/(2*sigma**2)) - (mu - x)**2*common*np.exp(-(mu - x)**2/(2*sigma**2))/sigma**2)/sigma**2) if diagonal else sum(2*amp*(mu-x)/(2*sigma**2)*np.exp(-((-mu+x)**2)/(2*sigma**2)) * 2*hess_amp*(hess_mu-x)/(hess_sigma**2)*np.exp(-((-hess_mu+x)**2)/(2*hess_sigma**2)))
                # us
                H[i-1, j+2] = sum(-2*(amp**2)/(sigma**5)*(mu-x)*((-mu+x)**2)*np.exp(-((-mu+x)**2)/(sigma**2)) - 2*amp*2*(mu-x)/(sigma**3)*np.exp(-((-mu+x)**2)/(2*sigma**2))*common + 2*amp*(mu-x)/(sigma**5)*((-mu+x)**2)*np.exp(-((-mu+x)**2)/(2*sigma**2))*common) if diagonal else sum(-hess_amp/(hess_sigma**3)*((-hess_mu+x)**2)*np.exp(-((-hess_mu+x)**2)/(2*hess_sigma**2)) * 2*amp*(mu-x)/(sigma**2)*np.exp(-((-mu+x)**2)/(2*sigma**2)))
                # sa
                H[i, j] = sum(2*amp*((-mu+x)**2)/(sigma**3)*np.exp(-((-mu+x)**2)/(sigma**2)) - 2*((-mu+x)**2)/(sigma**3)*np.exp(-((-mu+x)**2)/(2*sigma**2))*common) if diagonal else sum(2*amp*((-mu+x)**2)/(sigma**3)*np.exp(-((-mu+x)**2)/(2*sigma**2)) * np.exp(-((-hess_mu+x)**2)/(2*hess_sigma**2)))
                # su
                H[i, j+1] = sum(2*amp*(mu - x)*(-amp*(mu - x)**2*np.exp(-(mu - x)**2/sigma**2)/sigma**2 - 2*common*np.exp(-(mu - x)**2/(2*sigma**2)) + (mu - x)**2*common*np.exp(-(mu - x)**2/(2*sigma**2))/sigma**2)/sigma**3) if diagonal else sum(-2*hess_amp*(hess_mu-x)/(hess_sigma**2)*np.exp(-((-hess_mu+x)**2)/(2*hess_sigma**2)) * amp*(-mu+x)**2/(sigma**3)*np.exp(-((-mu+x)**2)/(2*sigma**2)))
                # ss
                H[i, j+2] = sum(2*(amp**2)*((-mu+x)**4)/(sigma**6)*np.exp(-((-mu+x)**2)/(sigma**2)) + 6*amp*((-mu+x)**2)/(sigma**4)*np.exp(-((-mu+x)**2)/(2*sigma**2))*common - 2*amp*((-mu+x)**4)/(sigma**6)*np.exp(-((-mu+x)**2)/(2*sigma**2))*common) if diagonal else sum(2*amp*((-mu+x)**2)/(sigma**3)*np.exp(-((-mu+x)**2)/(2*sigma**2)) * hess_amp*((-hess_mu+x)**2)/(hess_sigma**3)*np.exp(-((-hess_mu+x)**2)/(2*hess_sigma**2)))
    return H

# cpdef np.ndarray[FLOAT_t, ndim=1] bigauss_hess(np.ndarray[FLOAT_t, ndim=1] params, np.ndarray[FLOAT_t, ndim=1] x, np.ndarray[FLOAT_t, ndim=1] y):
#     cdef np.ndarray[FLOAT_t, ndim=1] common
#     cdef np.ndarray[FLOAT_t, ndim=2] H
#     H = np.zeros((params.shape[0], params.shape[0]))
#     common = -1*bigauss_ndim(x, params) + y
#     for i in xrange(0, params.shape[0], 4):
#         amp, mu, sigma, sigma2 = params[i:i+4]
#         for j in xrange(0, params.shape[0], 4):
#             hess_amp, hess_mu, hess_sigma, hess_sigma2 = params[j:j+4]
#             # the diagonal in general refers to the hessian with partial
#             # derivatives against itself (so a 3x3 or 4x4 area)
#             diagonal = i == j+2
#             lx = x[x<=mu]
#             left_common = common[x<=mu]
#             rx = x[x>mu]
#             right_common = common[x>mu]
#
#             hess_lx = x[x<=hess_mu]
#             hess_rx = x[x>hess_mu]
#
#             # aa left
#             H[i, j] += sum(2 * np.exp((-(-mu+lx)**2)/(sigma**2))) if diagonal else sum(2 * np.exp((-(-mu+lx)**2)/(2*sigma**2)) * np.exp((-(-hess_mu+lx)**2)/(2*hess_sigma**2)))
#             # aa right
#             H[i, j] += sum(2 * np.exp((-(-mu+rx)**2)/(sigma2**2))) if diagonal else sum(2 * np.exp((-(-mu+rx)**2)/(2*sigma2**2)) * np.exp((-(-hess_mu+rx)**2)/(2*hess_sigma2**2)))
#             # au left
#             H[i, j+1] += sum(-2*amp*(mu-lx)/(sigma**2) * np.exp((-(-mu+lx)**2)/(sigma**2)) + 2*(mu-lx)/(sigma**2)*left_common*np.exp(-((-mu+lx)**2)/(2*sigma**2))) if diagonal else sum(-2*hess_amp/(hess_sigma**2)*(hess_mu-lx)*np.exp(-((-mu+lx)**2)/(2*sigma**2))*np.exp(-((-hess_mu+lx)**2)/(2*hess_sigma**2)))
#             # au right
#             H[i, j+1] += sum(-2*amp*(mu-rx)/(sigma2**2) * np.exp((-(-mu+rx)**2)/(sigma2**2)) + 2*(mu-rx)/(sigma2**2)*right_common*np.exp(-((-mu+rx)**2)/(2*sigma2**2))) if diagonal else sum(-2*hess_amp/(hess_sigma2**2)*(hess_mu-rx)*np.exp(-((-mu+rx)**2)/(2*sigma2**2))*np.exp(-((-hess_mu+rx)**2)/(2*hess_sigma2**2)))
#             # as
#             H[i, j+2] += sum(2*amp*(-mu+lx)**2/(sigma**3)*np.exp(-(-mu+lx)**2/(sigma**2)) - 2*((-mu+lx)**2)/(sigma**3)*left_common*np.exp(-((-mu+lx)**2)/(2*sigma**2))) if diagonal else sum(2*hess_amp*(-hess_mu+lx)**2/(hess_sigma**3) * np.exp(-((-mu+lx)**2)/(2*sigma**2)) * np.exp(-((-hess_mu+lx)**2)/(2*hess_sigma**2)))
#             # as2
#             H[i, j+3] += sum(2*amp*(-mu+rx)**2/(sigma2**3)*np.exp(-(-mu+rx)**2/(sigma2**2)) - 2*((-mu+rx)**2)/(sigma2**3)*right_common*np.exp(-((-mu+rx)**2)/(2*sigma2**2))) if diagonal else sum(2*hess_amp*(-hess_mu+rx)**2/(hess_sigma2**3) * np.exp(-((-mu+rx)**2)/(2*sigma2**2)) * np.exp(-((-hess_mu+rx)**2)/(2*hess_sigma2**2)))
#
#             # ua left
#             H[i+1, j] += sum(-2*amp*(mu-lx)/(sigma**2)*np.exp(-((-mu+lx)**2)/(sigma**2)) + 2*(mu-lx)/(sigma**2)*left_common*np.exp(-((-mu+lx)**2)/(2*sigma**2))) if diagonal else sum(-2*amp*(mu-lx)/(sigma**2)*np.exp(-((-mu+lx)**2)/(2*sigma**2)) * np.exp(-((-hess_mu+lx)**2)/(2*hess_sigma**2)))
#             # ua right
#             H[i+1, j] += sum(-2*amp*(mu-rx)/(sigma2**2)*np.exp(-((-mu+rx)**2)/(sigma2**2)) + 2*(mu-rx)/(sigma2**2)*right_common*np.exp(-((-mu+rx)**2)/(2*sigma2**2))) if diagonal else sum(-2*amp*(mu-rx)/(sigma2**2)*np.exp(-((-mu+rx)**2)/(2*sigma2**2)) * np.exp(-((-hess_mu+rx)**2)/(2*hess_sigma2**2)))
#             # uu left
#             H[i+1, j+1] += sum(2*amp*(amp*(mu - lx)**2*np.exp(-(mu - lx)**2/sigma**2)/sigma**2 + left_common*np.exp(-(mu - lx)**2/(2*sigma**2)) - (mu - lx)**2*left_common*np.exp(-(mu - lx)**2/(2*sigma**2))/sigma**2)/sigma**2) if diagonal else sum(2*amp*(mu-lx)/(2*sigma**2)*np.exp(-((-mu+lx)**2)/(2*sigma**2)) * 2*hess_amp*(hess_mu-lx)/(hess_sigma**2)*np.exp(-((-hess_mu+lx)**2)/(2*hess_sigma**2)))
#             # uu right
#             H[i+1, j+1] += sum(2*amp*(amp*(mu - rx)**2*np.exp(-(mu - rx)**2/sigma2**2)/sigma2**2 + right_common*np.exp(-(mu - rx)**2/(2*sigma2**2)) - (mu - rx)**2*right_common*np.exp(-(mu - rx)**2/(2*sigma2**2))/sigma2**2)/sigma2**2) if diagonal else sum(2*amp*(mu-rx)/(2*sigma2**2)*np.exp(-((-mu+rx)**2)/(2*sigma2**2)) * 2*hess_amp*(hess_mu-rx)/(hess_sigma2**2)*np.exp(-((-hess_mu+rx)**2)/(2*hess_sigma2**2)))
#             # us
#             H[i+1, j+2] += sum(-2*(amp**2)/(sigma**5)*(mu-lx)*((-mu+lx)**2)*np.exp(-((-mu+lx)**2)/(sigma**2)) - 2*amp*2*(mu-lx)/(sigma**3)*np.exp(-((-mu+lx)**2)/(2*sigma**2))*left_common + 2*amp*(mu-lx)/(sigma**5)*((-mu+lx)**2)*np.exp(-((-mu+lx)**2)/(2*sigma**2))*left_common) if diagonal else sum(-hess_amp/(hess_sigma**3)*((-hess_mu+lx)**2)*np.exp(-((-hess_mu+lx)**2)/(2*hess_sigma**2)) * 2*amp*(mu-lx)/(sigma**2)*np.exp(-((-mu+lx)**2)/(2*sigma**2)))
#             # us2
#             H[i+1, j+3] += sum(-2*(amp**2)/(sigma2**5)*(mu-rx)*((-mu+rx)**2)*np.exp(-((-mu+rx)**2)/(sigma2**2)) - 2*amp*2*(mu-rx)/(sigma2**3)*np.exp(-((-mu+rx)**2)/(2*sigma2**2))*right_common + 2*amp*(mu-rx)/(sigma2**5)*((-mu+rx)**2)*np.exp(-((-mu+rx)**2)/(2*sigma2**2))*right_common) if diagonal else sum(-hess_amp/(hess_sigma2**3)*((-hess_mu+rx)**2)*np.exp(-((-hess_mu+rx)**2)/(2*hess_sigma2**2)) * 2*amp*(mu-rx)/(sigma2**2)*np.exp(-((-mu+rx)**2)/(2*sigma2**2)))
#
#             # sa left
#             H[i+2, j] += sum(2*amp*((-mu+lx)**2)/(sigma**3)*np.exp(-((-mu+lx)**2)/(sigma**2)) - 2*((-mu+lx)**2)/(sigma**3)*np.exp(-((-mu+lx)**2)/(2*sigma**2))*left_common) if diagonal else sum(2*amp*((-mu+lx)**2)/(sigma**3)*np.exp(-((-mu+lx)**2)/(2*sigma**2)) * np.exp(-((-hess_mu+lx)**2)/(2*hess_sigma**2)))
#             # sa right
#             H[i+2, j] += sum(2*amp*((-mu+rx)**2)/(sigma2**3)*np.exp(-((-mu+rx)**2)/(sigma2**2)) - 2*((-mu+rx)**2)/(sigma2**3)*np.exp(-((-mu+rx)**2)/(2*sigma2**2))*right_common) if diagonal else sum(2*amp*((-mu+rx)**2)/(sigma2**3)*np.exp(-((-mu+rx)**2)/(2*sigma2**2)) * np.exp(-((-hess_mu+rx)**2)/(2*hess_sigma2**2)))
#             # su left
#             H[i+2, j+1] += sum(2*amp*(mu - lx)*(-amp*(mu - lx)**2*np.exp(-(mu - lx)**2/sigma**2)/sigma**2 - 2*left_common*np.exp(-(mu - lx)**2/(2*sigma**2)) + (mu - lx)**2*left_common*np.exp(-(mu - lx)**2/(2*sigma**2))/sigma**2)/sigma**3) if diagonal else sum(-2*hess_amp*(hess_mu-lx)/(hess_sigma**2)*np.exp(-((-hess_mu+lx)**2)/(2*hess_sigma**2)) * amp*(-mu+lx)**2/(sigma**3)*np.exp(-((-mu+lx)**2)/(2*sigma**2)))
#             # su right
#             H[i+2, j+1] += sum(2*amp*(mu - rx)*(-amp*(mu - rx)**2*np.exp(-(mu - rx)**2/sigma2**2)/sigma2**2 - 2*right_common*np.exp(-(mu - rx)**2/(2*sigma2**2)) + (mu - rx)**2*right_common*np.exp(-(mu - rx)**2/(2*sigma2**2))/sigma2**2)/sigma2**3) if diagonal else sum(-2*hess_amp*(hess_mu-rx)/(hess_sigma2**2)*np.exp(-((-hess_mu+rx)**2)/(2*hess_sigma2**2)) * amp*(-mu+rx)**2/(sigma2**3)*np.exp(-((-mu+rx)**2)/(2*sigma2**2)))
#             # ss
#             H[i+2, j+2] += sum(2*(amp**2)*((-mu+lx)**4)/(sigma**6)*np.exp(-((-mu+lx)**2)/(sigma**2)) + 6*amp*((-mu+lx)**2)/(sigma**4)*np.exp(-((-mu+lx)**2)/(2*sigma**2))*left_common - 2*amp*((-mu+lx)**4)/(sigma**6)*np.exp(-((-mu+lx)**2)/(2*sigma**2))*left_common) if diagonal else sum(2*amp*((-mu+lx)**2)/(sigma**3)*np.exp(-((-mu+lx)**2)/(2*sigma**2)) * hess_amp*((-hess_mu+lx)**2)/(hess_sigma**3)*np.exp(-((-hess_mu+lx)**2)/(2*hess_sigma**2)))
#
#             # ss2
#             H[i+2, j+3] += 0# if diagonal else sum(2*amp*((-mu+rx)**2)/(sigma2**3)*np.exp(-((-mu+rx)**2)/(2*sigma2**2)) * hess_amp*((-hess_mu+rx)**2)/(hess_sigma2**3)*np.exp(-((-hess_mu+rx)**2)/(2*hess_sigma2**2)))
#
#             # s2a left
#             H[i+3, j] += sum(2*amp*((-mu+lx)**2)/(sigma**3)*np.exp(-((-mu+lx)**2)/(sigma**2)) - 2*((-mu+lx)**2)/(sigma**3)*np.exp(-((-mu+lx)**2)/(2*sigma**2))*left_common) if diagonal else sum(2*amp*((-mu+lx)**2)/(sigma**3)*np.exp(-((-mu+lx)**2)/(2*sigma**2)) * np.exp(-((-hess_mu+lx)**2)/(2*hess_sigma**2)))
#             # s2a right
#             H[i+3, j] += sum(2*amp*((-mu+rx)**2)/(sigma2**3)*np.exp(-((-mu+rx)**2)/(sigma2**2)) - 2*((-mu+rx)**2)/(sigma2**3)*np.exp(-((-mu+rx)**2)/(2*sigma2**2))*right_common) if diagonal else sum(2*amp*((-mu+rx)**2)/(sigma2**3)*np.exp(-((-mu+rx)**2)/(2*sigma2**2)) * np.exp(-((-hess_mu+rx)**2)/(2*hess_sigma2**2)))
#             # s2u left
#             H[i+3, j+1] += sum(2*amp*(mu - lx)*(-amp*(mu - lx)**2*np.exp(-(mu - lx)**2/sigma**2)/sigma**2 - 2*left_common*np.exp(-(mu - lx)**2/(2*sigma**2)) + (mu - lx)**2*left_common*np.exp(-(mu - lx)**2/(2*sigma**2))/sigma**2)/sigma**3) if diagonal else sum(-2*hess_amp*(hess_mu-lx)/(hess_sigma**2)*np.exp(-((-hess_mu+lx)**2)/(2*hess_sigma**2)) * amp*(-mu+lx)**2/(sigma**3)*np.exp(-((-mu+lx)**2)/(2*sigma**2)))
#             # s2u right
#             H[i+3, j+1] += sum(2*amp*(mu - rx)*(-amp*(mu - rx)**2*np.exp(-(mu - rx)**2/sigma2**2)/sigma2**2 - 2*right_common*np.exp(-(mu - rx)**2/(2*sigma2**2)) + (mu - rx)**2*right_common*np.exp(-(mu - rx)**2/(2*sigma2**2))/sigma2**2)/sigma2**3) if diagonal else sum(-2*hess_amp*(hess_mu-rx)/(hess_sigma2**2)*np.exp(-((-hess_mu+rx)**2)/(2*hess_sigma2**2)) * amp*(-mu+rx)**2/(sigma2**3)*np.exp(-((-mu+rx)**2)/(2*sigma2**2)))
#
#             # s2s
#             H[i+3, j+2] += 0# if diagonal else sum(2*amp*((-mu+x)**2)/(sigma**3)*np.exp(-((-mu+x)**2)/(2*sigma**2)) * hess_amp*((-hess_mu+x)**2)/(hess_sigma**3)*np.exp(-((-hess_mu+x)**2)/(2*hess_sigma**2)))
#
#             # s2s2
#             H[i+3, j+3] += sum(2*(amp**2)*((-mu+rx)**4)/(sigma2**6)*np.exp(-((-mu+rx)**2)/(sigma2**2)) + 6*amp*((-mu+rx)**2)/(sigma2**4)*np.exp(-((-mu+rx)**2)/(2*sigma2**2))*right_common - 2*amp*((-mu+rx)**4)/(sigma2**6)*np.exp(-((-mu+rx)**2)/(2*sigma2**2))*right_common) if diagonal else sum(2*amp*((-mu+rx)**2)/(sigma2**3)*np.exp(-((-mu+rx)**2)/(2*sigma2**2)) * hess_amp*((-hess_mu+rx)**2)/(hess_sigma2**3)*np.exp(-((-hess_mu+rx)**2)/(2*hess_sigma2**2)))
#     return H
