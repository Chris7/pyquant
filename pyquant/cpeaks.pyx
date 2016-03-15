import numpy as np
import sys
from copy import deepcopy
cimport numpy as np
cimport cython
ctypedef np.float_t FLOAT_t
from scipy import optimize, integrate
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

cpdef np.ndarray[FLOAT_t, ndim=2] bigauss_jac(np.ndarray[FLOAT_t, ndim=1] params, np.ndarray[FLOAT_t, ndim=1] x, np.ndarray[FLOAT_t, ndim=1] y):
    cdef np.ndarray[FLOAT_t, ndim=2] jac
    cdef np.ndarray[FLOAT_t, ndim=1] lx, ly, rx, ry
    cdef float amp, mu, stdl, stdr, sigma1, sigma2
    cdef int i
    jac = np.zeros((params.shape[0], 1))
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
            exp_term = np.exp(-((lx-mu)**2)/(2*sigma1**2))
            amp_exp_term = amp*exp_term
            prefix = 2*amp_exp_term
            # jac[i-3] += sum(prefix*(amp-ly*exp_term))
            jac[i-3, 0] += sum(-2*exp_term*(ly-amp_exp_term))
            jac[i-2, 0] += sum((-2*amp*(lx-mu)*exp_term*(ly-amp_exp_term))/(sigma1**2))
            jac[i-1, 0] += sum((-2*amp*((lx-mu)**2)*exp_term*(ly-amp_exp_term))/(sigma1**3))
            exp_term = np.exp(-((rx-mu)**2)/(2*sigma2**2))
            amp_exp_term = amp*exp_term
            prefix = 2*amp_exp_term
            # There is NO right side contribution to the jacobian of the amplitude because rx is defined as
            # x>mu, therefore anything by the right side of the bigaussian function does not change the amplitude
            jac[i-3, 0] += sum(-2*exp_term*(ry-amp_exp_term))
            jac[i-2, 0] += sum((-2*amp*(rx-mu)*exp_term*(ry-amp_exp_term))/sigma2**2)
            jac[i, 0] += sum((-2*amp*((rx-mu)**2)*exp_term*(ry-amp_exp_term))/(sigma2**3))
    return jac.transpose()

cpdef np.ndarray[FLOAT_t, ndim=2] gauss_jac(np.ndarray[FLOAT_t, ndim=1] params, np.ndarray[FLOAT_t, ndim=1] x, np.ndarray[FLOAT_t, ndim=1] y):
    cdef np.ndarray[FLOAT_t, ndim=2] jac
    cdef float amp, mu, sigma
    jac = np.zeros([len(params), 1])
    for i in xrange(len(params)):
        if i%3 == 0:
            amp = params[i]
        elif i%3 == 1:
            mu = params[i]
        elif i%3 == 2:
            sigma = params[i]
            exp_term = np.exp(-((x-mu)**2)/(2*sigma**2))
            amp_exp_term = amp*exp_term
            jac[i-2, 0] += sum(-2*exp_term*(y-amp_exp_term))
            jac[i-1, 0] += sum(-2*amp*(x-mu)*exp_term*(y-amp_exp_term)/(sigma**2))
            jac[i, 0] += sum(-2*amp*((x-mu)**2)*exp_term*(y-amp_exp_term)/(sigma**3))
    # print(params, jac)
    return jac.transpose()

cdef np.ndarray[FLOAT_t, ndim=1] bigauss(np.ndarray[FLOAT_t, ndim=1] x, float amp, float mu, float stdl, float stdr):
    cdef float sigma1 = stdl/1.177
    cdef float sigma2 = stdr/1.177
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
    # absolute deviation as our distance metric. Empirically found to give better results than
    # residual sum of squares for this data.
    cdef float residual = sum((ydata-data)**2)
    # cdef float fit, real, res
    # cdef float residual = 0
    # for i in range(len(ydata)):
    #     fit = data[i]
    #     real = ydata[i]
    #     res = (real-fit)**2
    #     if real > fit > 0.01:
    #         res = res*2*real/fit
    #     residual += res
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

cpdef basin_stepper(np.ndarray[FLOAT_t, ndim=1] args):
    args[::1] += 0.1
    args[::2] += 0.1
    args[::3] += 0.05
    args[::4] += 0.05
    return args

@cython.boundscheck(False)
cpdef tuple findAllPeaks(np.ndarray[FLOAT_t, ndim=1] xdata, np.ndarray[FLOAT_t, ndim=1] ydata_original,
                         float min_dist=0, filter=False, bigauss_fit=False, rt_peak=0.0, mrm=False,
                         int max_peaks=4, debug=False, peak_width_start=2, snr=0, amplitude_filter=0,
                         peak_width_end=4):
    cdef object fit_func, jacobian
    cdef np.ndarray[long, ndim=1] row_peaks, smaller_peaks, larger_peaks
    cdef np.ndarray[long, ndim=1] minima_array
    cdef list minima, fit_accuracy, smaller_minima, larger_minima, guess, bnds
    cdef dict peaks_found, final_peaks, peak_info
    cdef int peak_width, last_peak, next_peak, left, right, i, v, minima_index, left_stop, right_stop, right_stop_index
    cdef float peak_min, peak_max, rel_peak, average, variance, best_rss, rt_peak_val, minima_value
    cdef np.ndarray[FLOAT_t, ndim=1] peak_values, peak_indices, ydata, ydata_peaks, best_fit

    amplitude_filter /= ydata_original.max()
    ydata = ydata_original/ydata_original.max()

    ydata_peaks = np.copy(ydata)
    if snr != 0:
        ydata_peaks[ydata_peaks/np.std(ydata_peaks)<snr] = 0
    if amplitude_filter != 0:
        ydata_peaks[ydata_peaks<amplitude_filter] = 0
    if filter:
        if len(ydata) >= 5:
            ydata_peaks = convolve(ydata_peaks, kaiser(10, 12), mode='same')#gaussian_filter1d(convolve(ydata_peaks, kaiser(10, 14), mode='same'), 3, mode='constant')##gaussian_filter1d(ydata_peaks, 3, mode='constant')
            ydata_peaks[ydata_peaks<0] = 0
    ydata_peaks[np.isnan(ydata_peaks)] = 0
    mapper = interp1d(xdata, ydata_peaks)
    if rt_peak != 0:
        try:
            rt_peak_val = mapper(rt_peak)
        except ValueError:
            rt_peak_val = ydata_peaks[find_nearest_index(xdata, rt_peak)]
        ydata_peaks = np.where(ydata_peaks > rt_peak_val*0.9, ydata_peaks, 0)

    ydata_peaks /= ydata_peaks.max()

    peaks_found = {}
    if peak_width_start > peak_width_end:
        peak_width_start = peak_width_end
    peak_width = peak_width_start
    while peak_width <= peak_width_end:
        row_peaks = np.array(argrelmax(ydata_peaks, order=peak_width)[0], dtype=int)
        if not row_peaks.size:
            row_peaks = np.array([np.argmax(ydata)], dtype=int)
        if debug:
            sys.stderr.write('{}'.format(row_peaks))
        # Max peaks is to avoid spending a significant amount of time fitting bad data. It can lead to problems
        # if the user is searching the entire ms spectra because of the number of peaks possible to find
        if max_peaks != -1 and row_peaks.size > max_peaks:
            # pick the top n peaks for max_peaks
            # this selects the row peaks in ydata, reversed the sorting order (to be greatest to least), then
            # takes the number of peaks we allow and then sorts those peaks
            row_peaks = np.sort(row_peaks[np.argsort(ydata_peaks[row_peaks])[::-1]][:max_peaks])
            #peak_width_end += 1
            #peak_width += 1
            #continue
        if ydata_peaks.size:
            minima = np.where(ydata_peaks==0)[0].tolist()
        else:
            minima = []
        minima.extend([i for i in argrelmin(ydata_peaks, order=peak_width)[0] if i not in minima and i not in row_peaks])
        minima.sort()
        peaks_found[peak_width] = {'peaks': row_peaks, 'minima': minima}
        # if row_peaks.size > 1:
        #     peak_width_end += 1
        peak_width += 1
    # collapse identical orders
    if debug:
        sys.stderr.write('found: {}\n'.format(peaks_found))
    final_peaks = {}
    if peak_width_start == peak_width_end:
        final_peaks = peaks_found
    else:
        for peak_width in xrange(peak_width_start, peak_width_end):
            if debug:
                sys.stderr.write('checking {}\n'.format(peak_width))
            if peak_width not in peaks_found:
                continue
            smaller_peaks, smaller_minima = peaks_found[peak_width]['peaks'],peaks_found[peak_width]['minima']
            larger_peaks, larger_minima = peaks_found[peak_width+1]['peaks'],peaks_found[peak_width+1]['minima']
            if debug:
                sys.stderr.write('{}: {} ---- {}\n'.format(peak_width, smaller_peaks, larger_peaks))
            if set(smaller_peaks) == set(larger_peaks) and set(smaller_minima) ==  set(larger_minima):
                final_peaks[peak_width+1] = peaks_found[peak_width+1]
                if peak_width in final_peaks:
                    del final_peaks[peak_width]
            else:
                final_peaks[peak_width] = peaks_found[peak_width]
                if peak_width == peak_width_end-1:
                    final_peaks[peak_width+1] = peaks_found[peak_width+1]

    cdef tuple args
    cdef dict opts
    cdef list routines, results
    cdef str routine
    cdef object res, best_res
    best_res = 0
    cdef float bic, n, k, lowest_bic
    cdef float min_val

    fit_accuracy = []
    min_spacing = min(np.diff(xdata))/2
    peak_range = xdata[-1]-xdata[0]
    # initial bound setup
    initial_bounds = [(0, 1.01), (xdata[0], xdata[-1]), (min_spacing, peak_range)]
    if bigauss_fit:
        initial_bounds.extend([(min_spacing, peak_range)])
        # print(final_peaks)
    lowest_bic = 9999.
    if debug:
        sys.stderr.write('final peaks: {}\n'.format(final_peaks))
    for peak_width, peak_info in final_peaks.items():
        row_peaks = peak_info['peaks']
        if debug:
            print 'analyzing', row_peaks
        minima_array = np.array(peak_info['minima'], dtype=long)
        guess = []
        bnds = []
        last_peak = -1
        skip_peaks = set([])
        fitted_peaks = []
        for peak_num, peak_index in enumerate(row_peaks):
            if peak_index in skip_peaks:
                continue
            next_peak = len(xdata) if peak_index == row_peaks[-1] else row_peaks[peak_num+1]
            fitted_peaks.append(peak_index)
            rel_peak = ydata_peaks[peak_index]
            # find the points around it to estimate the std of the peak
            if minima_array.size:
                left = np.searchsorted(minima_array, peak_index)-1
                left_stop = np.searchsorted(minima_array, last_peak) if last_peak != -1 else -1
                if left == -1:
                    left = 0 if last_peak != -1 else last_peak+1
                elif left == left_stop:
                    if last_peak == -1:
                        left = 0
                    else:
                        left = minima_array[left]
                else:
                    for i in xrange(left, left_stop, -1):
                        minima_index = minima_array[i]
                        minima_value = ydata_peaks[minima_index]
                        if minima_value > rel_peak or minima_value < rel_peak*0.1 or ydata_peaks[minima_index-1]*0.9>minima_value:
                            if i == left:
                                left = minima_index
                            break
                        left = minima_index
                last_peak = peak_index
                right = np.searchsorted(minima_array, peak_index)
                right_stop = np.searchsorted(minima_array, next_peak)
                if False and right == right_stop:
                    right = minima_array[right]
                else:
                    for i in xrange(right, right_stop):
                        minima_index = minima_array[i]
                        minima_value = ydata_peaks[minima_index]
                        if minima_value > rel_peak or minima_value < rel_peak*0.1 or (minima_index+1 < ydata_peaks.size and ydata_peaks[minima_index+1]*0.9>minima_value):
                            if i == right:
                                right = minima_index
                            break
                        right = minima_index
                if right >= minima_array[-1]:
                    right = minima_array[-1]
                if right > next_peak:
                    right = next_peak
                if right < peak_index:
                    right = next_peak
                if right >= len(xdata):
                    bnds.extend([(rel_peak, 1.01), (xdata[left], xdata[-1]), (min_spacing, peak_range)])
                else:
                    bnds.extend([(rel_peak, 1.01), (xdata[left], xdata[right]), (min_spacing, peak_range)])
                if bigauss_fit:
                    bnds.extend([(min_spacing, peak_range)])
                peak_values = ydata[left:right]
                peak_indices = xdata[left:right]
            else:
                left = 0
                right = len(xdata)
                bnds.extend([(rel_peak, 1.01), (xdata[left], xdata[-1]), (min_spacing, peak_range)])
                if bigauss_fit:
                    bnds.extend([(min_spacing, peak_range)])
                peak_values = ydata[left:right]
                peak_indices = xdata[left:right]

            if debug:
                pass
                #print('bounds', peak_index, left, right, peak_values.tolist(), peak_indices.tolist(), bnds)

            if peak_values.any():
                average = np.average(peak_indices, weights=peak_values)
                variance = np.sqrt(np.average((peak_indices-average)**2, weights=peak_values))
                if variance == 0:
                    # we have a singular peak if variance == 0, so set the variance to half of the x/y spacing
                    if peak_index >= 1:
                        variance = np.abs(xdata[peak_index]-xdata[peak_index-1])
                    elif peak_index < len(xdata):
                        variance = np.abs(xdata[peak_index+1]-xdata[peak_index])
                    else:
                        # we have only 1 data point, most RT's fall into this width
                        variance = 0.05
            else:
                variance = 0.05
                average = xdata[peak_index]
            if variance is not None and variance < min_spacing:
                variance = min_spacing
            if variance is not None:
                if bigauss_fit:
                    guess.extend([rel_peak, average, variance, variance])
                else:
                    guess.extend([rel_peak, average, variance])
        if not guess:
            average = np.average(xdata, weights=ydata)
            variance = np.sqrt(np.average((xdata-average)**2, weights=ydata))
            if variance == 0:
                variance = 0.05
            guess = [max(ydata), average, variance]
            if bigauss_fit:
                guess.extend([variance])

        args = (xdata, ydata)
        opts = {'maxiter': 1000}
        fit_func = bigauss_func if bigauss_fit else gauss_func
        routines = ['SLSQP', 'TNC', 'L-BFGS-B']
        routine = routines.pop(0)
        if len(bnds) == 0:
            bnds = deepcopy(initial_bounds)
        jacobian = bigauss_jac if bigauss_fit else gauss_jac
        if debug:
            pass
            #print('guess and bnds', guess, bnds)
        results = [optimize.minimize(fit_func, guess, args, method=routine, bounds=bnds, options=opts, jac=jacobian)]
        while not results[-1].success and routines:
            routine = routines.pop(0)
            results.append(optimize.minimize(fit_func, guess, args, method=routine, bounds=bnds, options=opts, jac=jacobian))
            # print routine, res[-1]
        if results[-1].success:
            res = results[-1]
        else:
            res = sorted(results, key=attrgetter('fun'))[0]
        n = len(xdata)
        k = len(res.x)
        # this is actually incorrect, but works better...
        # bic = n*np.log(res.fun/n)+k+np.log(n)
        if bigauss_fit:
            bic = 2*k+2*np.log(res.fun/n)
        else:
            bic = res.fun
        res.bic = bic
        step_size = 4 if bigauss_fit else 3
        for index, value in enumerate(res.x[2::step_size]):
          if value < min_spacing:
            res.x[2+index*step_size] = min_spacing
        if bigauss_fit:
          for index, value in enumerate(res.x[3::step_size]):
            if value < min_spacing:
              res.x[3+index*step_size] = min_spacing
        # does this data contain our rt peak?
        res._contains_rt = False
        if rt_peak != 0:
            for i in xrange(1, res.x.size, 4 if bigauss_fit else 3):
                mean = res.x[i]
                lstd = res.x[i+1]
                if bigauss_fit:
                    rstd = res.x[i+2]
                else:
                    rstd = lstd
                if mean-lstd*2 < rt_peak < mean+rstd*2:
                    res._contains_rt = True

        if bic < lowest_bic or (getattr(best_res, '_contains_rt', False) and res._contains_rt == True):
            if debug:
                sys.stderr.write('{} < {}'.format(bic, lowest_bic))
            if res._contains_rt == False and best_res != 0 and best_res._contains_rt == True:
                continue
            best_fit = np.copy(res.x)
            best_res = res
            best_rss = res.fun
            lowest_bic = bic
        if debug:
            sys.stderr.write('{} - best: {}'.format(res, best_fit))

    return best_fit, best_rss

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

cdef inline int within_tolerance(list array, float tolerance):
    for i in array:
        if i[1] < tolerance:
            return 1
    return 0

def findMicro(np.ndarray[FLOAT_t, ndim=1] xdata, np.ndarray[FLOAT_t, ndim=1] ydata, pos, ppm=None,
              start_mz=None, calc_start_mz=None, isotope=0, spacing=0, quant_method='integrate', fragment_scan=False,
               centroid=False):
    """
        We want to find the boundaries of our isotopic clusters. Basically we search until our gradient
        changes, this assumes it's roughly gaussian and there is little interference
    """
    # find the edges within our tolerance
    cdef float tolerance
    tolerance = ppm
    cdef float offset, int_val, peak_mean
    offset = spacing*isotope
    cdef np.ndarray[FLOAT_t, ndim=1] df_empty_index = xdata[ydata==0]
    cdef int right, left
    cdef np.ndarray[FLOAT_t, ndim=1] new_x, new_y, lr
    if start_mz is None:
        start_mz = xdata[pos]
    fit = True
    if centroid:
        int_val = ydata[pos]
        left, right = pos-1, pos+1
        error = get_ppm(start_mz+offset, xdata[pos])
        fit = np.abs(error) < tolerance
        peak = [int_val, xdata[pos], 0]
    else:
        if df_empty_index.size == 0 or not (df_empty_index[0] < xdata[pos] < df_empty_index[-1]):
            left = 0
            right = xdata.size
        else:
            right = np.searchsorted(df_empty_index, xdata[pos])
            left = right-1
            left, right = (np.searchsorted(xdata, df_empty_index[left], side='left'),
                    np.searchsorted(xdata, df_empty_index[right]))
            right += 1
        new_x = xdata[left:right]
        new_y = ydata[left:right]
        if new_y.sum() == new_y.max():
            peak_mean = new_x[np.where(new_y>0)][0]
            peaks = (new_y.max(), peak_mean, 0)
            sorted_peaks = [(peaks, get_ppm(start_mz+offset, peak_mean))]
        else:
            peaks, peak_residuals = findAllPeaks(new_x, new_y, min_dist=(new_x[1]-new_x[0])*2.0, peak_width_start=1)
            sorted_peaks = sorted([(peaks[i*3:(i+1)*3], get_ppm(start_mz+offset, v)) for i,v in enumerate(peaks[1::3])], key=itemgetter(1))


        if fragment_scan == False and not within_tolerance(sorted_peaks, tolerance):
            if calc_start_mz is not None:
                sorted_peaks2 = sorted([(peaks[i*3:(i+1)*3], get_ppm(calc_start_mz+offset, v)) for i,v in enumerate(peaks[1::3])], key=itemgetter(1))
                if filter(lambda x: x[1]<tolerance, sorted_peaks2):
                    sorted_peaks = sorted_peaks2
                else:
                    fit = False
            else:
                fit = False

        peak = list(sorted_peaks[0][0])
        peak[0] *= new_y.max()

        int_val = gauss(new_x, peak[0], peak[1], peak[2]).sum()
        if not fit:
            pass
        error = sorted_peaks[0][1]
    ret_dict = {'int': int_val if fit or fragment_scan == True else 0, 'int2': int_val, 'bounds': (left, right), 'params': peak, 'error': error}
    return ret_dict

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

cpdef dict findEnvelope(np.ndarray[FLOAT_t, ndim=1] xdata, np.ndarray[FLOAT_t, ndim=1] ydata, measured_mz=None, theo_mz=None, max_mz=None, precursor_ppm=5, isotope_ppm=2.5, isotope_ppms=None, charge=2, debug=False,
                 isotope_offset=0, isotopologue_limit=-1, theo_dist=None, label=None, skip_isotopes=None, last_precursor=None, quant_method='integrate', reporter_mode=False, fragment_scan=False,
                 centroid=False, contaminant_search=True):
    # returns the envelope of isotopic peaks as well as micro envelopes  of each individual cluster
    cdef float spacing = NEUTRON/float(charge)
    cdef float tolerance, precursor_tolerance
    start_mz = measured_mz if isotope_offset == 0 else measured_mz+isotope_offset*NEUTRON/float(charge)
    initial_mz = start_mz
    if max_mz is not None:
        max_mz = max_mz-spacing*0.9 if isotope_offset == 0 else max_mz+isotope_offset*NEUTRON*0.9/float(charge)
    if isotope_ppms is None:
        isotope_ppms = {}
    tolerance = isotope_ppms.get(0, precursor_ppm)/1000000.0
    env_dict, micro_dict, ppm_dict = OrderedDict(),OrderedDict(),OrderedDict()
    empty_dict = {'envelope': env_dict, 'micro_envelopes': micro_dict, 'ppms': ppm_dict}

    cdef np.ndarray[FLOAT_t] non_empty = xdata[ydata>0]
    if len(non_empty) == 0:
        if debug:
            print('data is empty')
        return empty_dict
    first_mz = find_nearest(non_empty, start_mz)
    attempts = 0


    isotope_index = 0
    use_theo = False
    # This is purposefully verbose to be more explicit
    if reporter_mode == False and fragment_scan == False:
        while get_ppm(start_mz, first_mz) > tolerance:
            # let's try using our theoretical mass
            first_mz = find_nearest(non_empty, theo_mz)
            if get_ppm(theo_mz, first_mz) > tolerance:
                # let's check our last boundary
                if last_precursor is not None:
                    first_mz = find_nearest(non_empty, last_precursor)
                    if get_ppm(last_precursor, first_mz) > tolerance:
                        # repeat all of that for the next isotopic index
                        start_mz += spacing
                        initial_mz += spacing
                        theo_mz += spacing
                        last_precursor += spacing
                        isotope_index += 1
                    else:
                        start_mz = last_precursor
                        break
                else:
                    start_mz += spacing
                    theo_mz += spacing
                    initial_mz += spacing
                    isotope_index += 1
            else:
                use_theo = True
                break
            tolerance = isotope_ppms.get(isotope_index, isotope_ppm)/1000000.0
            if isotope_index == 2 or (max_mz is not None and first_mz >= max_mz):
                if debug:
                    print('unable to find start ion')
                return empty_dict

    precursor_tolerance = tolerance

    isotope_index += isotope_offset
    start_index = find_nearest_index(xdata, first_mz)
    start_info = findMicro(xdata, ydata, start_index, ppm=tolerance, start_mz=start_mz, calc_start_mz=theo_mz,
     quant_method=quant_method, fragment_scan=fragment_scan, centroid=centroid)
    start_error = start_info['error']

    if 'params' in start_info:
        if fragment_scan == False and start_info['error'] > tolerance:
            start = last_precursor if last_precursor is not None else theo_mz if use_theo else start_mz
        else:
            start = start_info['params'][1]
    else:
        if debug:
            print('empty start info', start_info)
        return empty_dict

    valid_locations2 = OrderedDict()
    valid_locations2[isotope_index] = [(0, start, find_nearest_index(non_empty, start))]
    contaminant_bounds = {}
    contaminant_int = 0.

    if not reporter_mode and (isotopologue_limit == -1 or len(valid_locations2) < isotopologue_limit):
        isotope_index += 1
        pos = find_nearest_index(non_empty, start)+1
        offset = isotope_index*spacing
        df_len = non_empty.shape[0]
        last_displacement = None
        valid_locations = []

        # check for contaminant at doubly and triply charged positions to see if we're in another ion's peak
        if contaminant_search:
            for i in xrange(2, 4):
                closest_contaminant = find_nearest(non_empty, start-NEUTRON/float(i))
                closest_contaminant_index = find_nearest_index(xdata, closest_contaminant)
                contaminant_bounds = findMicro(xdata, ydata, closest_contaminant_index, ppm=precursor_tolerance,
                                         calc_start_mz=start, start_mz=start, isotope=-1, spacing=NEUTRON/float(i),
                                          quant_method=quant_method, centroid=centroid)
                if contaminant_bounds.get('int', 0) > contaminant_int:
                    contaminant_int = contaminant_bounds.get('int', 0.)

        # set the tolerance for isotopes
        tolerance = isotope_ppms.get(isotope_index, isotope_ppm)/1000000.0

        while pos < df_len:
            # search for the ppm error until it rises again, we select the minima and if this minima is
            # outside our ppm error, we stop the expansion of our isotopic cluster
            current_loc = non_empty[pos]
            if max_mz is not None and current_loc >= max_mz:
                if not valid_locations:
                    break
                displacement = last_displacement+tolerance if last_displacement is not None else tolerance*2
            else:
                displacement = get_ppm(start+offset, current_loc)
            # if debug:
            #     print pos, start, current_loc, displacement, last_displacement, displacement > last_displacement, last_displacement < tolerance, isotope_index, offset
            # because the peak location may be between two readings, we use a very tolerance search here and enforce the ppm at the peak fitting stage.
            if displacement < tolerance*5:
                valid_locations.append((displacement, current_loc, pos))
            if last_displacement is not None:
                if valid_locations and displacement > last_displacement:
                    # pick the peak closest to our error tolerance
                    valid_locations2[isotope_index] = valid_locations
                    isotope_index += 1
                    tolerance = isotope_ppms.get(isotope_index, isotope_ppm)/1000000.0
                    offset = spacing*isotope_index
                    displacement = get_ppm(start+offset, current_loc)
                    valid_locations = []
                    if isotopologue_limit != -1 and (len(valid_locations2) >= isotopologue_limit):
                        break
                elif displacement > last_displacement and not valid_locations:
                    break
            last_displacement = displacement
            pos += 1

    #combine any overlapping micro envelopes
    #final_micros = self.merge_list(micro_dict)
    valid_keys = sorted(set(valid_locations2.keys()).intersection(theo_dist.keys() if theo_dist is not None else valid_locations2.keys()))
    # This attempts to use a diophantine equation to match the clusters to the theoretical distribution of isotopes
    #valid_vals = [j[1] for i in valid_keys for j in valid_locations2[i]]
    #valid_theor = pd.Series([theo_dist[i] for i in valid_keys])
    #valid_theor = valid_theor/valid_theor.max()
    #best_locations = sorted(looper(selected=valid_vals, df=df, theo=valid_theor), key=itemgetter(0))[0][1]
    best_locations = [sorted(valid_locations2[i], key=itemgetter(0))[0] for i in valid_keys]


    for index, isotope_index in enumerate(valid_keys):
        if skip_isotopes is not None and isotope_index in skip_isotopes:
            continue
        _, _, empty_index = best_locations[index]
        micro_index = find_nearest_index(xdata, non_empty[empty_index])
        if ydata[micro_index] == 0:
            micro_index += 1
        if ydata[micro_index] == 0:
            micro_index -= 2
        # if micro_index == 0:
        #     pass
        isotope_tolerance = isotope_ppms.get(isotope_index, isotope_ppm)/1000000.0
        micro_bounds = findMicro(xdata, ydata, micro_index, ppm=precursor_tolerance if isotope_index == 0 else isotope_tolerance,
                                 calc_start_mz=start, start_mz=start_mz, isotope=isotope_index, spacing=spacing, quant_method=quant_method, centroid=centroid)
        if isotope_index == 0:
            micro_bounds['error'] = start_error

        micro_dict[isotope_index] = micro_bounds
        env_dict[isotope_index] = micro_index
        ppm_dict[isotope_index] = micro_bounds.get('error')

    # in all cases, the envelope is going to be either monotonically decreasing, or a parabola (-x^2)
    isotope_pattern = [(isotope_index, isotope_dict['int']) for isotope_index, isotope_dict in micro_dict.items()]
    # Empirically, it's been found that enforcing the theoretical distribution on a per ms1 scan basis leads to
    # significant increases in variance for the XIC
    # if theo_dist is not None and len(isotope_pattern) >= 2:
    #     pass
        # ref_iso = -1
        # ref_int = 0.
        # theo_int = 0.
        # for i,(isotope_index, isotope_intensity) in enumerate(isotope_pattern):
        #     if isotope_index == ref_iso:
        #         continue
        #     # if isotope_intensity == 0 and ref_iso > 0:
        #     #     print isotope_index
        #     #     print ref_iso, ref_int
        #     #     print theo_dist
        #     #     print micro_dict[isotope_index]
        #     if ref_int == 0 and isotope_intensity > 0:
        #         if contaminant_int > 1 and float(isotope_intensity)/contaminant_int < 1:
        #             env_dict.pop(isotope_index)
        #             micro_dict.pop(isotope_index)
        #             ppm_dict.pop(isotope_index)
        #         else:
        #             ref_iso = isotope_index
        #             ref_int = float(isotope_intensity)
        #             theo_int = float(theo_dist[ref_iso])
        #     elif isotope_intensity > 0:
        #         theo_ratio = theo_int/theo_dist[isotope_index]
        #         data_ratio = ref_int/isotope_intensity
        #         if contaminant_int > 1 and ref_int/contaminant_int < 1:
        #             env_dict.pop(isotope_index)
        #             micro_dict.pop(isotope_index)
        #             ppm_dict.pop(isotope_index)
        #         elif np.abs(np.log2(data_ratio/theo_ratio)) > 0.5:
        #             if debug:
        #                 print('pattern1 loss', label, isotope_index, theo_ratio, data_ratio, micro_dict[isotope_index])
        #             env_dict.pop(isotope_index)
        #             micro_dict.pop(isotope_index)
        #             ppm_dict.pop(isotope_index)
    if contaminant_int > 1:
        for i,(isotope_index, isotope_intensity) in enumerate(isotope_pattern):
            if contaminant_int > isotope_intensity:
                if debug:
                    print('contaminant loss', label)
                env_dict.pop(isotope_index)
                micro_dict.pop(isotope_index)
                ppm_dict.pop(isotope_index)
    # rebuild the pattern after contaminants are removed
    isotope_pattern = [(isotope_index, isotope_dict['int']) for isotope_index, isotope_dict in micro_dict.items()]
    if theo_dist is None:
        # are we monotonically decreasing?
        remove = False
        if len(isotope_pattern) > 2:
            # check if the 2nd isotope is smaller than the first. This is a classical case looking like:
            #
            #  |
            #  |  |
            #  |  |  |
            #  |  |  |  |

            if isotope_pattern[1][1] < isotope_pattern[0][1]:
                # we are, check this trend holds and remove isotopes it fails for
                for i,j in zip(isotope_pattern, isotope_pattern[1:]):
                    if j[1]*0.9 > i[1]:
                        # the pattern broke, remove isotopes beyond this point
                        remove = True
                    if remove:
                        if debug:
                            print('pattern2.1 loss', label, j[0], isotope_pattern)
                        env_dict.pop(j[0])
                        micro_dict.pop(j[0])
                        ppm_dict.pop(j[0])

            # check if the 2nd isotope is larger than the first. This is a case looking like:
            #
            #
            #     |  |
            #     |  |
            #  |  |  |  |

            elif isotope_pattern[1][1] > isotope_pattern[0][1]:
                shift = False
                for i,j in zip(isotope_pattern, isotope_pattern[1:]):
                    if shift and j[1]*0.9 > i[1]:
                        remove = True
                    elif shift is False and j[1] < i[1]*0.9:
                        if shift:
                            remove = True
                        else:
                            shift = True
                    if remove:
                        if debug:
                            print('pattern2.2 loss', label, j[0], isotope_pattern)
                        env_dict.pop(j[0])
                        micro_dict.pop(j[0])
                        ppm_dict.pop(j[0])

    return {'envelope': env_dict, 'micro_envelopes': micro_dict, 'ppms': ppm_dict}
