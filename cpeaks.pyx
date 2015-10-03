import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport isnan, fabs
ctypedef np.float_t FLOAT_t
from scipy import optimize, integrate
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import argrelmax, argrelmin, convolve, kaiser
from operator import itemgetter
from collections import OrderedDict
from pythomics.proteomics.config import NEUTRON

cdef int within_bounds(np.ndarray[FLOAT_t, ndim=1] res, list bnds):
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
    cdef float residual = sum(np.abs(ydata-data)**2)
    return residual

cdef np.ndarray[FLOAT_t, ndim=1] bigauss(np.ndarray[FLOAT_t, ndim=1] x, float amp, float mu, float stdl, float stdr):
    cdef float sigma1 = stdl/1.177
    cdef float m1 = np.sqrt(2*np.pi)*sigma1*amp
    cdef float sigma2 = stdr/1.177
    cdef float m2 = np.sqrt(2*np.pi)*sigma2*amp
    cdef np.ndarray[FLOAT_t, ndim=1] lx = x[x<=mu]
    cdef np.ndarray[FLOAT_t, ndim=1] left = m1/(np.sqrt(2*np.pi)*sigma1)*np.exp(-(lx-mu)**2/(2*sigma1**2))
    cdef np.ndarray[FLOAT_t, ndim=1] rx = x[x>mu]
    cdef np.ndarray[FLOAT_t, ndim=1] right = m2/(np.sqrt(2*np.pi)*sigma2)*np.exp(-(rx-mu)**2/(2*sigma2**2))
    cdef np.ndarray[FLOAT_t, ndim=1] y = np.concatenate([left, right], axis=1)
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
    if any([isnan(i) for i in guess]):
        return np.inf
    cdef np.ndarray[FLOAT_t, ndim=1] data = bigauss_ndim(xdata, guess)
    # absolute deviation as our distance metric. Empirically found to give better results than
    # residual sum of squares for this data.
    cdef float residual = sum(np.abs(ydata-data)**2)
    return residual

cpdef np.ndarray[FLOAT_t] fixedMeanFit(np.ndarray[FLOAT_t, ndim=1] xdata, np.ndarray[FLOAT_t, ndim=1] ydata,
                                       int peak_index=1, debug=False):
    cdef float rel_peak = ydata[peak_index]
    cdef float peak_loc = xdata[peak_index]
    cdef int peak_left, peak_right
    cdef float peak_min, peak_max, average, variance

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
    bnds = [(rel_peak*0.75, 1.0) if rel_peak > 0 else (0.0, 1.0), (xdata[0], xdata[-1]), (0.0, fabs(peak_loc-xdata[0])), (0.0, fabs(xdata[-1]-peak_loc))]
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
        print peak_loc
        print xdata.tolist()
        print ydata.tolist()
        print bnds
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

cpdef tuple findAllPeaks(np.ndarray[FLOAT_t, ndim=1] xdata, np.ndarray[FLOAT_t, ndim=1] ydata_original, float min_dist=0, filter=False, bigauss_fit=False):
    cdef object fit_func
    cdef np.ndarray[long] row_peaks, smaller_peaks, larger_peaks
    cdef list minima, fit_accuracy, smaller_minima, larger_minima, guess, bnds
    cdef dict peaks_found, final_peaks, peak_info
    cdef int peak_width, last_peak, next_peak, left, right, i, v
    cdef float peak_min, peak_max, rel_peak, average, variance
    cdef np.ndarray[FLOAT_t] peak_values, peak_indices, ydata

    ydata = ydata_original/ydata_original.max()
    cdef np.ndarray[FLOAT_t, ndim=1] ydata_peaks
    ydata_peaks = np.copy(ydata)
    if filter:
        if len(ydata) >= 5:
            ydata_peaks = gaussian_filter1d(ydata_peaks, 3, mode='constant')
            ydata_peaks[ydata_peaks<0] = 0
    peaks_found = {}
    for peak_width in xrange(1,4):
        row_peaks = argrelmax(ydata_peaks, order=peak_width)[0]
        if not row_peaks.size:
            row_peaks = np.array([np.argmax(ydata)], dtype=int)
        if ydata_peaks.size:
            minima = [i for i,v in enumerate(ydata_peaks) if v == 0]
        else:
            minima = []
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

    cdef tuple args
    cdef dict opts
    cdef list routines, best_fits
    cdef str routine
    cdef object res
    cdef int n, k
    cdef float bic

    fit_accuracy = []
    for peak_width, peak_info in final_peaks.items():
        row_peaks = peak_info['peaks']
        minima = peak_info['minima']
        guess = []
        bnds = []
        last_peak = -1
        for peak_num, peak_index in enumerate(row_peaks):
            next_peak = len(xdata)-1 if peak_index == row_peaks[-1] else row_peaks[peak_num+1]
            peak_min, peak_max = xdata[peak_index]-0.2, xdata[peak_index]+0.2

            peak_min = xdata[0] if peak_min < xdata[0] else peak_min
            peak_max = xdata[-1] if peak_max > xdata[-1] else peak_max
            rel_peak = ydata[peak_index]/sum(ydata[row_peaks])
            bnds.extend([(rel_peak, 1), (peak_min, peak_max), (0.0001, peak_max-peak_min)])
            if bigauss_fit:
                bnds.extend([(0.0001, peak_max-peak_min)])
            # find the points around it to estimate the std of the peak
            left = 0
            for i,v in enumerate(minima):
                if v >= peak_index:
                    if i != 0:
                        left = minima[i-1]
                    break
                left = v
            if last_peak != -1 and left < last_peak:
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
        res = optimize.minimize(fit_func, guess, args, method=routine, bounds=bnds, options=opts)
        while not res.success and routines:
            routine = routines.pop(0)
            res = optimize.minimize(fit_func, guess, args, method=routine, bounds=bnds, options=opts)
        n = len(xdata)
        k = len(res.x)
        bic = n*np.log(res.fun/n)+k+np.log(n)
        res.bic = bic
        fit_accuracy.append((peak_width, bic, res.x, xdata[row_peaks]))
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
    return fabs(theoretical-observed)/theoretical

cdef inline int within_tolerance(list array, float tolerance):
    for i in array:
        if i[1] < tolerance:
            return 1
    return 0

def findMicro(np.ndarray[FLOAT_t, ndim=1] xdata, np.ndarray[FLOAT_t, ndim=1] ydata, pos, ppm=None,
              start_mz=None, calc_start_mz=None, isotope=0, spacing=0, quant_method='integrate'):
    """
        We want to find the boundaries of our isotopic clusters. Basically we search until our gradient
        changes, this assumes it's roughly gaussian and there is little interference
    """
    # find the edges within our tolerance
    cdef float tolerance
    tolerance = ppm
    cdef float offset, int_val
    offset = spacing*isotope
    cdef np.ndarray[FLOAT_t, ndim=1] df_empty_index = xdata[ydata==0]
    cdef int right, left
    cdef np.ndarray[FLOAT_t, ndim=1] new_x, new_y, lr
    if len(df_empty_index) == 0:
        right = pos+1
        left = pos
        peak = (ydata[pos], xdata[pos], 0.01)
        ret_dict = {'int': ydata[pos], 'bounds': (left, right), 'params': peak, 'error': 0}
    else:
        right = np.searchsorted(df_empty_index, xdata[pos])
        left = right-1
        left, right = (np.searchsorted(xdata, df_empty_index[left], side='left'),
                np.searchsorted(xdata, df_empty_index[right]))
        right += 1
        new_x = xdata[left:right]
        new_y = ydata[left:right]
        peaks, peak_centers = findAllPeaks(new_x, new_y, min_dist=(new_x[1]-new_x[0])*2.0)
        if start_mz is None:
            start_mz = xdata[pos]

        sorted_peaks = sorted([(peaks[i*3:(i+1)*3], get_ppm(start_mz+offset, v)) for i,v in enumerate(peaks[1::3])], key=itemgetter(1))
        fit = True

        if not within_tolerance(sorted_peaks, tolerance):
            if calc_start_mz is not None:
                sorted_peaks2 = sorted([(peaks[i*3:(i+1)*3], get_ppm(calc_start_mz+offset, v)) for i,v in enumerate(peaks[1::3])], key=itemgetter(1))
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

        lr = np.linspace(peak_gauss[1]-peak_gauss[2]*4, peak_gauss[1]+peak_gauss[2]*4, 1000)
        left_peak, right_peak = peak[1]-peak[2]*2, peak[1]+peak[2]*2
        int_val = integrate.simps(gauss(lr, peak_gauss[0], peak_gauss[1], peak_gauss[2]), x=lr) if quant_method == 'integrate' else ydata[(xdata > left_peak) & (xdata < right_peak)].sum()
        if not fit:
            pass
        ret_dict = {'int': int_val if fit else 0, 'bounds': (left, right), 'params': peak, 'error': sorted_peaks[0][1]}

    return ret_dict

def find_nearest(np.ndarray[FLOAT_t, ndim=1] array, value):
    return array[find_nearest_index(array, value)]

def find_nearest_index(np.ndarray[FLOAT_t, ndim=1] array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx == 0:
        return 0
    elif idx == len(array):
        return -1
    elif fabs(value - array[idx-1]) < fabs(value - array[idx]):
        return idx-1
    else:
        return idx

def findEnvelope(np.ndarray[FLOAT_t, ndim=1] xdata, np.ndarray[FLOAT_t, ndim=1] ydata, measured_mz=None, theo_mz=None, max_mz=None, precursor_ppm=5, isotope_ppm=2.5, isotope_ppms=None, charge=2, debug=False,
                 isotope_offset=0, isotopologue_limit=-1, theo_dist=None, label=None, skip_isotopes=None, last_precursor=None, quant_method='integrate', reporter_mode=False):
    # returns the envelope of isotopic peaks as well as micro envelopes  of each individual cluster
    cdef float spacing = NEUTRON/float(charge)
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
        return empty_dict
    first_mz = find_nearest(non_empty, start_mz)
    attempts = 0


    isotope_index = 0
    use_theo = False
    # This is purposefully verbose to be more explicit
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
            return empty_dict

    isotope_index += isotope_offset
    start_index = find_nearest_index(xdata, first_mz)
    start_info = findMicro(xdata, ydata, start_index, ppm=tolerance, start_mz=start_mz, calc_start_mz=theo_mz, quant_method=quant_method)
    start_error = start_info['error']

    if 'params' in start_info:
        if start_info['error'] > tolerance:
            start = last_precursor if last_precursor is not None else theo_mz if use_theo else start_mz
        else:
            start = start_info['params'][1]
    else:
        return empty_dict

    valid_locations2 = OrderedDict()
    valid_locations2[isotope_index] = [(0, start, find_nearest_index(non_empty, start))]

    if not reporter_mode and (isotopologue_limit == -1 or len(valid_locations2) < isotopologue_limit):
        isotope_index += 1
        pos = find_nearest_index(non_empty, start)+1
        offset = isotope_index*spacing
        df_len = non_empty.shape[0]
        last_displacement = None
        valid_locations = []
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
            if debug:
                print pos, start, current_loc, displacement, last_displacement, displacement > last_displacement, last_displacement < tolerance, isotope_index, offset
            if displacement < tolerance:
                valid_locations.append((displacement, current_loc, pos))
            if valid_locations and displacement > last_displacement:
                # pick the largest peak within our error tolerance
                valid_locations2[isotope_index] = valid_locations
                isotope_index += 1
                tolerance = isotope_ppms.get(isotope_index, isotope_ppm)/1000000.0
                offset = spacing*isotope_index
                displacement = get_ppm(start+offset, current_loc)
                valid_locations = []
                if len(valid_locations2) >= isotopologue_limit:
                    break
            elif last_displacement is not None and displacement > last_displacement and not valid_locations:
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
        precursor_tolerance = isotope_ppms.get(0, precursor_ppm)/1000000.0
        micro_bounds = findMicro(xdata, ydata, micro_index, ppm=precursor_tolerance if isotope_index == 0 else isotope_tolerance,
                                 calc_start_mz=start, start_mz=start_mz, isotope=isotope_index, spacing=spacing, quant_method=quant_method)
        if isotope_index == 0:
            micro_bounds['error'] = start_error

        micro_dict[isotope_index] = micro_bounds
        env_dict[isotope_index] = micro_index
        ppm_dict[isotope_index] = micro_bounds.get('error')

    # if label == 'Heavy':
    #     print micro_dict


    # in all cases, the envelope is going to be either monotonically decreasing, or a parabola (-x^2)
    isotope_pattern = [(isotope_index, isotope_dict['int']) for isotope_index, isotope_dict in micro_dict.items()]
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
                    env_dict.pop(j[0])
                    micro_dict.pop(j[0])
                    ppm_dict.pop(j[0])

    return {'envelope': env_dict, 'micro_envelopes': micro_dict, 'ppms': ppm_dict}
