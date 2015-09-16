import pyximport; pyximport.install()
import numpy as np
cimport numpy as np
ctypedef np.float_t FLOAT_t

def gauss(np.array[FLOAT_t, ndim=1]x, float amp, float mu, float std):
    return amp*np.exp(-(x - mu)**2/(2*std**2))

def gauss_ndim(np.array[FLOAT_t, ndim=1] xdata, *args):
    amps, mus, sigmas = args[::3], args[1::3], args[2::3]
    data = np.zeros(len(xdata))
    for amp, mu, sigma in zip(amps, mus, sigmas):
        data += gauss(xdata, amp, mu, sigma)
    return data

def gauss_func(guess, *args):
    xdata, ydata = args
    data = gauss_ndim(xdata, *guess)
    # absolute deviation as our distance metric. Empirically found to give better results than
    # residual sum of squares for this data.
    return sum(np.abs(ydata-data)**2)

def bigauss(np.array[FLOAT_t, ndim=1] x, float amp, float mu, float stdl, float stdr):
    float sigma1 = stdl/1.177
    float m1 = np.sqrt(2*np.pi)*sigma1*amp
    float sigma2 = stdr/1.177
    float m2 = np.sqrt(2*np.pi)*sigma2*amp
    #left side
    if isinstance(x, float):
        x = np.array([x])
    lx = x[x<=mu]
    int left = m1/(np.sqrt(2*np.pi)*sigma1)*np.exp(-(lx-mu)**2/(2*sigma1**2))
    rx = x[x>mu]
    int right = m2/(np.sqrt(2*np.pi)*sigma2)*np.exp(-(rx-mu)**2/(2*sigma2**2))
    return np.concatenate([left, right], axis=1)

def bigauss_ndim(np.array[FLOAT_t, ndim=1] xdata, *args):
    amps, mus, sigmasl, sigmasr = args[::4], args[1::4], args[2::4], args[3::4]
    data = np.zeros(len(xdata))
    for amp, mu, sigma1, sigma2 in zip(amps, mus, sigmasl, sigmasr):
        data += bigauss(xdata, amp, mu, sigma1, sigma2)
    return data

def bigauss_func(guess, *args):
    xdata, ydata = args
    if any([pd.isnull(i) for i in guess]):
        return np.inf
    data = bigauss_ndim(xdata, *guess)
    # absolute deviation as our distance metric. Empirically found to give better results than
    # residual sum of squares for this data.
    return sum(np.abs(ydata-data)**2)

def fixedMeanFit(np.array[FLOAT_t, ndim=1] xdata, np.array[FLOAT_t, ndim=1] ydata, peak_index=None, debug=False):

    ydata /= ydata.max()

    rel_peak = ydata[peak_index]
    peak_loc = xdata[peak_index]
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
    bnds = [(rel_peak*0.75, 1) if rel_peak > 0 else (0, 1), (xdata[0], xdata[-1]), (0, peak_loc-xdata[0]), (0, xdata[-1]-peak_loc)]
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

def findAllPeaks(values, min_dist=0, filter=False, bigauss_fit=False):
    xdata = values.index.values.astype(float)
    ydata = values.fillna(0).values.astype(float)

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
        if np.array_equal(smaller_peaks, larger_peaks) and np.array_equal(smaller_minima, larger_minima):
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

def findPeak(y, srt):
    # check our SNR, if it's low, lessen our window
    left_offset = 1
    right_offset = 2
    lsrt = srt-left_offset if srt-left_offset > 0 else 0
    rsrt = srt+right_offset if srt+right_offset < len(y) else len(y)
    peak = y[srt]
    left = 0
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