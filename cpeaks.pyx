# cython: linetrace=True
import pyximport; pyximport.install()
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport isnan
ctypedef np.float_t FLOAT_t
from scipy import optimize, integrate
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import argrelmax, argrelmin
from operator import itemgetter

cdef int within_bounds(res, bnds):
    for i,j in zip(res.x, bnds):
        if j[0] is not None and i < j[0]:
            return False
        if j[1] is not None and i > j[1]:
            return False
    return True

def gauss(np.ndarray[FLOAT_t, ndim=1] x, float amp, float mu, float std):
    cdef np.ndarray[FLOAT_t, ndim=1] y = amp*np.exp(-(x - mu)**2/(2*std**2))
    return y

def gauss_ndim(np.ndarray[FLOAT_t, ndim=1] xdata, *args):
    amps, mus, sigmas = args[::3], args[1::3], args[2::3]
    cdef np.ndarray[FLOAT_t, ndim=1] data = np.zeros(len(xdata))
    for amp, mu, sigma in zip(amps, mus, sigmas):
        data += gauss(xdata, amp, mu, sigma)
    return data

def gauss_func(guess, np.ndarray[FLOAT_t, ndim=1] xdata, np.ndarray[FLOAT_t, ndim=1] ydata):
    cdef np.ndarray[FLOAT_t, ndim=1] data = gauss_ndim(xdata, *guess)
    # absolute deviation as our distance metric. Empirically found to give better results than
    # residual sum of squares for this data.
    cdef float residual = sum(np.abs(ydata-data)**2)
    return residual

def bigauss(np.ndarray[FLOAT_t, ndim=1] x, float amp, float mu, float stdl, float stdr):
    cdef float sigma1 = stdl/1.177
    cdef float m1 = np.sqrt(2*np.pi)*sigma1*amp
    cdef float sigma2 = stdr/1.177
    cdef float m2 = np.sqrt(2*np.pi)*sigma2*amp
    #left side
    if isinstance(x, float):
        x = np.ndarray([x])
    cdef np.ndarray[FLOAT_t, ndim=1] lx = x[x<=mu]
    cdef np.ndarray[FLOAT_t, ndim=1] left = m1/(np.sqrt(2*np.pi)*sigma1)*np.exp(-(lx-mu)**2/(2*sigma1**2))
    cdef np.ndarray[FLOAT_t, ndim=1] rx = x[x>mu]
    cdef np.ndarray[FLOAT_t, ndim=1] right = m2/(np.sqrt(2*np.pi)*sigma2)*np.exp(-(rx-mu)**2/(2*sigma2**2))
    cdef np.ndarray[FLOAT_t, ndim=1] y = np.concatenate([left, right], axis=1)
    return y

def bigauss_ndim(np.ndarray[FLOAT_t, ndim=1] xdata, *args):
    amps, mus, sigmasl, sigmasr = args[::4], args[1::4], args[2::4], args[3::4]
    cdef np.ndarray[FLOAT_t, ndim=1] data = np.zeros(len(xdata))
    for amp, mu, sigma1, sigma2 in zip(amps, mus, sigmasl, sigmasr):
        data += bigauss(xdata, amp, mu, sigma1, sigma2)
    return data

def bigauss_func(guess, *args):
    cdef np.ndarray[FLOAT_t, ndim=1] xdata = args[0]
    cdef np.ndarray[FLOAT_t, ndim=1] ydata = args[1]
    if any([isnan(i) for i in guess]):
        return np.inf
    cdef np.ndarray[FLOAT_t, ndim=1] data = bigauss_ndim(xdata, *guess)
    # absolute deviation as our distance metric. Empirically found to give better results than
    # residual sum of squares for this data.
    cdef float residual = sum(np.abs(ydata-data)**2)
    return residual

def fixedMeanFit(np.ndarray[FLOAT_t, ndim=1] xdata, np.ndarray[FLOAT_t, ndim=1] ydata, peak_index=None, debug=False):

    ydata /= ydata.max()

    cdef float rel_peak = ydata[peak_index]
    cdef float peak_loc = xdata[peak_index]
    # print xdata.tolist()
    # print ydata.tolist()

    from scipy.signal import convolve, bartlett, hamming, kaiser
    peak_left, peak_right = findPeak(convolve(ydata, kaiser(10, 14), mode='same'), peak_index)
    #print peak_left, peak_right
    #print xdata.tolist()
    #print ydata.tolist()
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
    bnds = [(rel_peak*0.75, 1.0) if rel_peak > 0 else (0.0, 1.0), (xdata[0], xdata[-1]), (0.0, peak_loc-xdata[0]), (0.0, xdata[-1]-peak_loc)]
    cdef float average = np.average(xdata, weights=ydata)
    cdef float variance = np.sqrt(np.average((xdata-average)**2, weights=ydata))
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
    guess = [rel_peak, peak_loc, variance, variance]
    #print guess
    #print bnds
    # if values.name > 729.36 and values.name < 731:
    #     print guess, bnds
    args = (xdata, ydata)
    base_opts = {'maxiter': 1000, 'ftol': 1e-20}
    routines = [('SLSQP', base_opts), ('TNC', base_opts), ('L-BFGS-B', base_opts)]
    routine, opts = routines.pop(0)
    results = [optimize.minimize(bigauss_func, guess, args, bounds=bnds, method=routine, options=opts, tol=1e-20)]#, jac=gauss_jac)]
    # if values.name > 729.36 and values.name < 731:
    while routines:
        # if values.name > 729.36 and values.name < 731:
        # if debug:
        #     print results[-1]
        routine, opts = routines.pop(0)
        results.append(optimize.minimize(bigauss_func, guess, args, bounds=bnds, method=routine, options=opts, tol=1e-20))#, jac=gauss_jac)
    n = len(xdata)
    # if not results[-1].success:
    res = sorted([i for i in results if within_bounds(i, bnds)], key=lambda x: x.fun)[0]
    # else:
    #     res = results[-1]
    k = len(res.x)
    bic = n*np.log(res.fun/n)+k+np.log(n)
    res.bic = bic
    return res

def findAllPeaks(np.ndarray[FLOAT_t, ndim=1] xdata, np.ndarray[FLOAT_t, ndim=1] ydata, min_dist=0, filter=False, bigauss_fit=False):
    ydata /= ydata.max()
    ydata_peaks = np.copy(ydata)
    if filter:
        if len(ydata) >= 5:
            # ydata_peaks = values.replace([0], np.nan).interpolate(method='index').values
            ydata_peaks = gaussian_filter1d(ydata_peaks, 3, mode='constant')
            ydata_peaks[ydata_peaks<0] = 0
    peaks_found = {}
    for peak_width in xrange(1,4):
        row_peaks = argrelmax(ydata_peaks, order=peak_width)[0]
        if not row_peaks.any():
            row_peaks = [np.argmax(ydata)]
        minima = [i for i,v in enumerate(ydata_peaks) if v == 0]
        minima.extend([i for i in argrelmin(ydata_peaks, order=peak_width)[0] if i not in minima])
        minima.sort()
        peaks_found[peak_width] = {'peaks': row_peaks, 'minima': minima}
    # collapse identical orders
    final_peaks = {}
    for peak_width in xrange(1, 3):
        if peak_width == len(peaks_found):
            final_peaks[peak_width] = peaks_found[peak_width]
            continue
        smaller_peaks, smaller_minima = peaks_found[peak_width]['peaks'],peaks_found[peak_width]['minima']
        larger_peaks, larger_minima = peaks_found[peak_width+1]['peaks'],peaks_found[peak_width+1]['minima']
        if set(smaller_peaks) == set(larger_peaks) and set(smaller_minima) ==  set(larger_minima):
            final_peaks[peak_width+1] = peaks_found[peak_width+1]
            if peak_width in final_peaks:
                del final_peaks[peak_width]
        else:
            final_peaks[peak_width] = peaks_found[peak_width]
    fit_accuracy = []
    for peak_width, peak_info in final_peaks.items():
        row_peaks = peak_info['peaks']
        minima = peak_info['minima']
        guess = []
        bnds = []
        last_peak = None
        for peak_num, peak_index in enumerate(row_peaks):
            next_peak = len(xdata)-1 if peak_index == row_peaks[-1] else row_peaks[peak_num+1]
            peak_min, peak_max = xdata[peak_index]-0.2, xdata[peak_index]+0.2

            peak_min = xdata[0] if peak_min < xdata[0] else peak_min
            peak_max = xdata[-1] if peak_max > xdata[-1] else peak_max
            rel_peak = ydata[peak_index]/sum(ydata[row_peaks])
            std_bounds = (0.0001, peak_max-peak_min)
            bnds.extend([(rel_peak, 1), (peak_min, peak_max), std_bounds])
            if bigauss_fit:
                bnds.extend([std_bounds])
            # find the points around it to estimate the std of the peak
            left = 0
            for i,v in enumerate(minima):
                if v >= peak_index:
                    if i != 0:
                        left = minima[i-1]
                    break
                left = v
            if last_peak is not None and left < last_peak:
                left = last_peak
            last_peak = peak_index
            right = len(xdata)
            for right in minima:
                if right > peak_index or right >= next_peak:
                    if right < len(xdata) and right != next_peak:
                        right += 1
                    break
            if right > next_peak:
                right = next_peak
            if right < peak_index:
                right = next_peak
            peak_values = ydata[left:right]
            peak_indices = xdata[left:right]
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
            if variance is not None:
                guess.extend([ydata[peak_index], average, variance])
                if bigauss_fit:
                    guess.extend([variance])

        if not guess:
            average = np.average(xdata, weights=ydata)
            variance = np.sqrt(np.average((xdata-average)**2, weights=ydata))
            if variance == 0:
                variance = 0.05
            guess = [max(ydata), np.argmax(ydata), variance]
            if bigauss_fit:
                guess.extend([variance])

        args = (xdata, ydata)
        opts = {'maxiter': 1000}
        fit_func = bigauss_func if bigauss_fit else gauss_func
        routines = ['SLSQP', 'TNC', 'L-BFGS-B', 'SLSQP']
        routine = routines.pop(0)
        res = optimize.minimize(fit_func, guess, args, method=routine, bounds=bnds, options=opts)#, jac=gauss_jac)
        while not res.success and routines:
            routine = routines.pop(0)
            res = optimize.minimize(fit_func, guess, args, method=routine, bounds=bnds, options=opts)#, jac=gauss_jac)
        n = len(xdata)
        k = len(res.x)
        bic = n*np.log(res.fun/n)+k+np.log(n)
        res.bic = bic
        fit_accuracy.append((peak_width, bic, res, xdata[row_peaks]))
    # we want to maximize our BIC given our definition
    best_fits = sorted(fit_accuracy, key=itemgetter(1,0), reverse=True)
    return best_fits[0][2:]

cdef tuple findPeak(np.ndarray[FLOAT_t, ndim=1] y, int srt):
    # check our SNR, if it's low, lessen our window
    cdef int left_offset = 1
    cdef int right_offset = 2
    cdef int lsrt = srt-left_offset if srt-left_offset > 0 else 0
    cdef int rsrt = srt+right_offset if srt+right_offset < len(y) else len(y)
    cdef float peak = y[srt]
    cdef int left = 0
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

cdef float get_ppm(float theoretical, float observed):
    return np.abs(theoretical-observed)/theoretical

def findMicro(np.ndarray[FLOAT_t, ndim=1] xdata, np.ndarray[FLOAT_t, ndim=1] ydata, pos, ppm=None,
              start_mz=None, calc_start_mz=None, isotope=0, spacing=0, quant_method='integrate'):
    """
        We want to find the boundaries of our isotopic clusters. Basically we search until our gradient
        changes, this assumes it's roughly gaussian and there is little interference
    """
    # find the edges within our tolerance
    tolerance = ppm
    cdef float offset = spacing*isotope
    cdef np.ndarray[FLOAT_t, ndim=1] df_empty_index = xdata[ydata==0]
    cdef int right = np.searchsorted(df_empty_index, xdata[pos])
    cdef int left = right-1
    left, right = (np.searchsorted(xdata, df_empty_index[left], side='left'),
            np.searchsorted(xdata, df_empty_index[right]))
    right += 1
    cdef np.ndarray[FLOAT_t, ndim=1] new_x = xdata[left:right]
    cdef np.ndarray[FLOAT_t, ndim=1] new_y = ydata[left:right]
    peaks, peak_centers = findAllPeaks(new_x, new_y, min_dist=(new_x[1]-new_x[0])*2)
    if start_mz is None:
        start_mz = xdata[pos]

    # new logic is nm
    sorted_peaks = sorted([(peaks.x[i*3:(i+1)*3], get_ppm(start_mz+offset, v)) for i,v in enumerate(peaks.x[1::3])], key=lambda x: x[1])
    fit = True

    if not filter(lambda x: x[1]<tolerance, sorted_peaks):
        if calc_start_mz is not None:
            sorted_peaks2 = sorted([(peaks.x[i*3:(i+1)*3], get_ppm(calc_start_mz+offset, v)) for i,v in enumerate(peaks.x[1::3])], key=lambda x: x[1])
            if filter(lambda x: x[1]<tolerance, sorted_peaks2):
                sorted_peaks = sorted_peaks2
            else:
                fit = False
        else:
            fit = False

    peak = sorted_peaks[0][0]
    # interpolate our mean/std to a linear range
    from scipy.interpolate import interp1d
    mapper = interp1d(new_x, range(len(new_x)))
    try:
        mu = mapper(peak[1])
    except:
        print 'mu', sorted_peaks, peak, new_x
        return {'int': 0, 'error': np.inf}
    try:
        std = mapper(new_x[0]+np.abs(peak[2]))-mapper(new_x[0])
    except:
        print 'std', sorted_peaks, peak, new_x
        return {'int': 0, 'error': np.inf}
    peak_gauss = (peak[0]*new_y.max(), mu, std)
    peak[0] *= new_y.max()

    cdef np.ndarray[FLOAT_t, ndim=1] lr = np.linspace(peak_gauss[1]-peak_gauss[2]*4, peak_gauss[1]+peak_gauss[2]*4, 1000)
    left_peak, right_peak = peak[1]-peak[2]*2, peak[1]+peak[2]*2
    cdef float int_val = integrate.simps(gauss(lr, *peak_gauss), x=lr)# if quant_method == 'integrate' else y[(y.index > left_peak) & (y.index < right_peak)].sum()
    if not fit:
        pass

    return {'int': int_val if fit else 0, 'bounds': (left, right), 'params': peak, 'error': sorted_peaks[0][1]}