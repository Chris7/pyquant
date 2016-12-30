import os
from collections import defaultdict

import numpy as np
import six

if os.environ.get('PYQUANT_DEV', False) == 'True':
    try:
        import pyximport; pyximport.install(setup_args={'include_dirs': np.get_include()}, reload_support=True)
    except:
        import traceback
        traceback.print_exc()
        pass

from pyquant.cpeaks import *
from .utils import select_window, divide_peaks, argrelextrema

if six.PY3:
    xrange = range

_epsilon = np.sqrt(np.finfo(float).eps)

def findEnvelope(xdata, ydata, measured_mz=None, theo_mz=None, max_mz=None, precursor_ppm=5, isotope_ppm=2.5,
                 isotope_ppms=None, charge=2, debug=False, isotope_offset=0, isotopologue_limit=-1,
                 theo_dist=None, label=None, skip_isotopes=None, last_precursor=None, quant_method='integrate',
                 reporter_mode=False, fragment_scan=False, centroid=False, contaminant_search=True):
    # returns the envelope of isotopic peaks as well as micro envelopes  of each individual cluster
    spacing = NEUTRON / float(charge)
    start_mz = measured_mz if isotope_offset == 0 else measured_mz + isotope_offset * NEUTRON / float(charge)
    initial_mz = start_mz
    if max_mz is not None:
        max_mz = max_mz - spacing * 0.9 if isotope_offset == 0 else max_mz + isotope_offset * NEUTRON * 0.9 / float(
            charge)
    if isotope_ppms is None:
        isotope_ppms = {}
    tolerance = isotope_ppms.get(0, precursor_ppm) / 1000000.0
    env_dict, micro_dict, ppm_dict = OrderedDict(), OrderedDict(), OrderedDict()
    empty_dict = {'envelope': env_dict, 'micro_envelopes': micro_dict, 'ppms': ppm_dict}

    non_empty = xdata[ydata > 0]
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
            tolerance = isotope_ppms.get(isotope_index, isotope_ppm) / 1000000.0
            if isotope_index == 2 or (max_mz is not None and first_mz >= max_mz):
                if debug:
                    print('unable to find start ion')
                return empty_dict

    precursor_tolerance = tolerance

    isotope_index += isotope_offset
    start_index = find_nearest_index(xdata, first_mz)
    start_info = findMicro(xdata, ydata, start_index, ppm=tolerance, start_mz=start_mz, calc_start_mz=theo_mz,
                           quant_method=quant_method, reporter_mode=reporter_mode, fragment_scan=fragment_scan, centroid=centroid)
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
        pos = find_nearest_index(non_empty, start) + 1
        offset = isotope_index * spacing
        df_len = non_empty.shape[0]
        last_displacement = None
        valid_locations = []

        # check for contaminant at doubly and triply charged positions to see if we're in another ion's peak
        if contaminant_search:
            for i in xrange(2, 4):
                closest_contaminant = find_nearest(non_empty, start - NEUTRON / float(i))
                closest_contaminant_index = find_nearest_index(xdata, closest_contaminant)
                contaminant_bounds = findMicro(xdata, ydata, closest_contaminant_index, ppm=precursor_tolerance,
                                               calc_start_mz=start, start_mz=start, isotope=-1,
                                               spacing=NEUTRON / float(i),
                                               quant_method=quant_method, centroid=centroid)
                if contaminant_bounds.get('int', 0) > contaminant_int:
                    contaminant_int = contaminant_bounds.get('int', 0.)

        # set the tolerance for isotopes
        tolerance = isotope_ppms.get(isotope_index, isotope_ppm) / 1000000.0

        while pos < df_len:
            # search for the ppm error until it rises again, we select the minima and if this minima is
            # outside our ppm error, we stop the expansion of our isotopic cluster
            current_loc = non_empty[pos]
            if max_mz is not None and current_loc >= max_mz:
                if not valid_locations:
                    break
                displacement = last_displacement + tolerance if last_displacement is not None else tolerance * 2
            else:
                displacement = get_ppm(start + offset, current_loc)
            # because the peak location may be between two readings, we use a very tolerance search here and enforce the ppm at the peak fitting stage.
            if displacement < tolerance * 5:
                valid_locations.append((displacement, current_loc, pos))
            if last_displacement is not None:
                if valid_locations and displacement > last_displacement:
                    # pick the peak closest to our error tolerance
                    valid_locations2[isotope_index] = valid_locations
                    isotope_index += 1
                    tolerance = isotope_ppms.get(isotope_index, isotope_ppm) / 1000000.0
                    offset = spacing * isotope_index
                    displacement = get_ppm(start + offset, current_loc)
                    valid_locations = []
                    if isotopologue_limit != -1 and (len(valid_locations2) >= isotopologue_limit):
                        break
                elif displacement > last_displacement and not valid_locations:
                    break
            last_displacement = displacement
            pos += 1

    # combine any overlapping micro envelopes
    valid_keys = sorted(set(valid_locations2.keys()).intersection(
        theo_dist.keys() if theo_dist is not None else valid_locations2.keys()))
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
        isotope_tolerance = isotope_ppms.get(isotope_index, isotope_ppm) / 1000000.0
        micro_bounds = findMicro(xdata, ydata, micro_index,
                                 ppm=precursor_tolerance if isotope_index == 0 else isotope_tolerance,
                                 calc_start_mz=start, start_mz=start_mz, isotope=isotope_index, spacing=spacing,
                                 quant_method=quant_method, centroid=centroid)
        if isotope_index == 0:
            micro_bounds['error'] = start_error

        micro_dict[isotope_index] = micro_bounds
        env_dict[isotope_index] = micro_index
        ppm_dict[isotope_index] = micro_bounds.get('error')

    # in all cases, the envelope is going to be either monotonically decreasing, or a parabola (-x^2)
    isotope_pattern = [(isotope_index, isotope_dict['int']) for isotope_index, isotope_dict in micro_dict.items()]
    # Empirically, it's been found that enforcing the theoretical distribution on a per ms1 scan basis leads to
    # significant increases in variance for the XIC, so don't do it here
    if contaminant_int > 1:
        for i, (isotope_index, isotope_intensity) in enumerate(isotope_pattern):
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
                for i, j in zip(isotope_pattern, isotope_pattern[1:]):
                    if j[1] * 0.9 > i[1]:
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
                for i, j in zip(isotope_pattern, isotope_pattern[1:]):
                    if shift and j[1] * 0.9 > i[1]:
                        remove = True
                    elif shift is False and j[1] < i[1] * 0.9:
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


def findAllPeaks(xdata, ydata_original, min_dist=0, method=None, local_filter_size=0, filter=False, bigauss_fit=False, rt_peak=0.0, mrm=False,
                 max_peaks=4, debug=False, peak_width_start=2, snr=0, zscore=0, amplitude_filter=0, peak_width_end=4,
                 baseline_correction=False, rescale=True):
    amplitude_filter /= ydata_original.max()
    ydata = ydata_original / ydata_original.max()
    ydata_peaks = np.copy(ydata)
    if filter:
        if len(ydata) >= 5:
            ydata_peaks = convolve(ydata_peaks, kaiser(10, 12), mode='same')
            ydata_peaks[ydata_peaks < 0] = 0
    ydata_peaks[np.isnan(ydata_peaks)] = 0
    ydata_peaks_std = np.std(ydata_peaks)
    ydata_peaks_median = np.median(ydata_peaks)
    if rt_peak != 0:
        mapper = interp1d(xdata, ydata_peaks)
        try:
            rt_peak_val = mapper(rt_peak)
        except ValueError:
            rt_peak_val = ydata_peaks[find_nearest_index(xdata, rt_peak)]
        ydata_peaks = np.where(ydata_peaks > rt_peak_val * 0.9, ydata_peaks, 0)

    ydata_peaks /= ydata_peaks.max()

    peaks_found = {}
    if peak_width_start > peak_width_end:
        peak_width_end = peak_width_start + 1
    peak_width = peak_width_start
    while peak_width <= peak_width_end:
        row_peaks = np.array(argrelextrema(ydata_peaks, np.greater, order=peak_width)[0], dtype=int)
        if not row_peaks.size:
            row_peaks = np.array([np.argmax(ydata)], dtype=int)
        if debug:
            sys.stderr.write('peak indices: {}\n'.format(row_peaks))

        if snr != 0 or zscore != 0:
            if local_filter_size:
                new_peaks = []
                lost_peaks = {}
                for row_peak in row_peaks:
                    selection = select_window(ydata_peaks, row_peak, local_filter_size)
                    local_std = np.std(selection)
                    local_snr = ydata_peaks[row_peak] / local_std
                    local_zscore = (ydata_peaks[row_peak] - np.median(selection)) / local_std
                    add_peak = (snr == 0 or local_snr > snr) and \
                          (zscore == 0 or  local_zscore >= zscore)
                    if add_peak:
                        new_peaks.append(row_peak)
                    elif debug:
                        lost_peaks[row_peak] = {'snr': local_snr, 'zs': local_zscore}
                if debug:
                    sys.stderr.write('{} peaks lost to filtering\n{}\n'.format(len(row_peaks)-len(new_peaks), lost_peaks))
                row_peaks = np.array(new_peaks, dtype=int)
            else:
                if debug and snr:
                    sys.stderr.write('{} peaks lost to SNR\n'.format(sum(ydata_peaks[row_peaks] / ydata_peaks_std < snr)))
                if debug and zscore:
                    sys.stderr.write(
                        '{} peaks lost to zscore\n'.format(sum((ydata_peaks[row_peaks] - ydata_peaks_median) / ydata_peaks_std < zscore)))
                if snr:
                    row_peaks = row_peaks[ydata_peaks[row_peaks] / ydata_peaks_std >= snr]
                if zscore:
                    row_peaks = row_peaks[(ydata_peaks[row_peaks] - ydata_peaks_median) / ydata_peaks_std >= zscore]


        if amplitude_filter != 0:
            if debug:
                sys.stderr.write('{} peaks lost to amp filter\n{}\n'.format(sum(ydata[row_peaks] < amplitude_filter), row_peaks[ydata[row_peaks] < amplitude_filter]))
            row_peaks = row_peaks[ydata[row_peaks] >= amplitude_filter]
        # Max peaks is to avoid spending a significant amount of time fitting bad data. It can lead to problems
        # if the user is searching the entire ms spectra because of the number of peaks possible to find
        if max_peaks != -1 and row_peaks.size > max_peaks:
            # pick the top n peaks for max_peaks
            if rt_peak:
                # If the user specified a retention time as a guide, select the n peaks closest
                row_peaks = np.sort(np.abs(xdata[row_peaks]-rt_peak)[:max_peaks])
            else:
                # this selects the row peaks in ydata, reversed the sorting order (to be greatest to least), then
                # takes the number of peaks we allow and then sorts those peaks
                row_peaks = np.sort(row_peaks[np.argsort(ydata_peaks[row_peaks])[::-1]][:max_peaks])
            # peak_width_end += 1
            # peak_width += 1
            # continue
        if ydata_peaks.size:
            minima = np.where(ydata_peaks == 0)[0].tolist()
        else:
            minima = []
        minima.extend(
            [i for i in argrelextrema(ydata_peaks, np.less, order=peak_width)[0] if i not in minima and i not in row_peaks])
        minima.sort()
        peaks_found[peak_width] = {'peaks': row_peaks, 'minima': minima}
        # if row_peaks.size > 1:
        #     peak_width_end += 1
        peak_width += 1

    # Next, for fitting multiple peaks, we want to divide up the space so we are not fitting peaks that
    # have no chance of actually impacting one another.
    chunks = divide_peaks(ydata_peaks)
    if not chunks.any() or chunks[-1] != len(ydata_peaks):
        chunks = np.hstack((chunks, len(ydata_peaks)))

    # Now that we've found our peaks and breakpoints between peaks, we can obliterate part of ydata_peaks
    if snr != 0 and not local_filter_size:
        ydata_peaks[ydata_peaks / ydata_peaks_std < snr] = 0
    if zscore != 0 and not local_filter_size:
        ydata_peaks[(ydata_peaks - ydata_peaks_median) / ydata_peaks_std < zscore] = 0
    if amplitude_filter != 0:
        ydata_peaks[ydata_peaks < amplitude_filter] = 0
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
            smaller_peaks, smaller_minima = peaks_found[peak_width]['peaks'], peaks_found[peak_width]['minima']
            larger_peaks, larger_minima = peaks_found[peak_width + 1]['peaks'], peaks_found[peak_width + 1]['minima']
            if debug:
                sys.stderr.write('{}: {} ---- {}\n'.format(peak_width, smaller_peaks, larger_peaks))
            if set(smaller_peaks) == set(larger_peaks) and set(smaller_minima) == set(larger_minima):
                final_peaks[peak_width + 1] = peaks_found[peak_width + 1]
                if peak_width in final_peaks:
                    del final_peaks[peak_width]
            else:
                final_peaks[peak_width] = peaks_found[peak_width]
                if peak_width == peak_width_end - 1:
                    final_peaks[peak_width + 1] = peaks_found[peak_width + 1]


    fit_accuracy = []
    step_size = 4 if bigauss_fit else 3
    if baseline_correction:
        step_size += 2
    min_spacing = min(np.diff(xdata)) / 2
    peak_range = xdata[-1] - xdata[0]
    # initial bound setup
    initial_bounds = [(0, 1.01), (xdata[0], xdata[-1]), (min_spacing, peak_range)]
    if bigauss_fit:
        initial_bounds.extend([(min_spacing, peak_range)])
    if baseline_correction:
        initial_bounds.extend([(None, None), (None, None)])
        # print(final_peaks)
    if debug:
        sys.stderr.write('final peaks: {}\n'.format(final_peaks))

    fitted_segments = defaultdict(list)
    for peak_width, peak_info in final_peaks.items():
        row_peaks = peak_info['peaks']
        minima_array = np.array(peak_info['minima'], dtype=np.long)
        guess = []
        bnds = []
        last_peak = -1
        skip_peaks = set([])
        fitted_peaks = []
        for peak_num, peak_index in enumerate(row_peaks):
            if peak_index in skip_peaks:
                continue
            next_peak = len(xdata) if peak_index == row_peaks[-1] else row_peaks[peak_num + 1]
            fitted_peaks.append(peak_index)
            rel_peak = ydata_peaks[peak_index]
            # bounds for fitting the peak mean
            peak_left = xdata[peak_index - 1]
            peak_right = xdata[peak_index + 1] if peak_index+1 < len(xdata) else xdata[-1]
            # find the points around it to estimate the std of the peak
            if minima_array.size:
                left = np.searchsorted(minima_array, peak_index) - 1
                left_stop = np.searchsorted(minima_array, last_peak) if last_peak != -1 else -1
                if left == -1:
                    left = 0 if last_peak != -1 else last_peak + 1
                elif left == left_stop:
                    if last_peak == -1:
                        left = 0
                    else:
                        left = minima_array[left]
                elif left_stop > left:
                    # We are at the last minima, set our left bound to the last peak if it is greater than
                    # the left minima, otherwise set to the left minima
                    minima_index = minima_array[left]
                    left = last_peak if last_peak > minima_index else minima_index
                else:
                    for i in xrange(left, left_stop, -1):
                        minima_index = minima_array[i]
                        minima_value = ydata_peaks[minima_index]
                        if minima_value > rel_peak or minima_value < rel_peak * 0.1 or ydata_peaks[
                                    minima_index - 1] * 0.9 > minima_value or (
                                    peak_index - i > 3 and minima_value > rel_peak * 0.25):
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
                        if minima_value > rel_peak or minima_value < rel_peak * 0.1 or (
                                    minima_index + 1 < ydata_peaks.size and ydata_peaks[
                                minima_index + 1] * 0.9 > minima_value):
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
                if right >= len(xdata)-1:
                    right = len(xdata)-1
                bnds.extend([(rel_peak, 1.01), (xdata[left], xdata[right]) if baseline_correction else (peak_left, peak_right), (min_spacing, peak_range)])
                if bigauss_fit:
                    bnds.extend([(min_spacing, peak_range)])
                if baseline_correction:
                    bnds.extend([(None, None), (None, None)])
                peak_values = ydata[left:right]
                peak_indices = xdata[left:right]
            else:
                left = 0
                right = len(xdata)-1
                bnds.extend([(rel_peak, 1.01), (peak_left, peak_right), (min_spacing, peak_range)])
                if bigauss_fit:
                    bnds.extend([(min_spacing, peak_range)])
                if baseline_correction:
                    bnds.extend([(None, None), (None, None)])
                peak_values = ydata[left:right+1]
                peak_indices = xdata[left:right+1]

            if debug:
                print('bounds', peak_index, left, right, peak_values.tolist(), peak_indices.tolist(), bnds)

            if peak_values.any():
                average = np.average(peak_indices, weights=peak_values)
                variance = np.sqrt(np.average((peak_indices - average) ** 2, weights=peak_values))
                if variance == 0:
                    # we have a singular peak if variance == 0, so set the variance to half of the x/y spacing
                    if peak_index >= 1:
                        variance = np.abs(xdata[peak_index] - xdata[peak_index - 1])
                    elif peak_index < len(xdata):
                        variance = np.abs(xdata[peak_index + 1] - xdata[peak_index])
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
                if baseline_correction:
                    slope = (ydata[right]-ydata[left])/(xdata[right]-xdata[left])
                    intercept = ((ydata[right]-slope*xdata[right])+(ydata[left]-slope*xdata[left]))/2
                    guess.extend([slope, intercept])
        if not guess:
            average = np.average(xdata, weights=ydata)
            variance = np.sqrt(np.average((xdata - average) ** 2, weights=ydata))
            if variance == 0:
                variance = 0.05
            guess = [max(ydata), average, variance]
            if bigauss_fit:
                guess.extend([variance])
            if baseline_correction:
                slope = (ydata[-1] - ydata[0]) / (xdata[-1] - xdata[0])
                intercept = ((ydata[-1] - slope * xdata[-1]) + (ydata[0] - slope * xdata[0])) / 2
                guess.extend([slope, intercept])


        # Now that we have estimated the parameters for fitting all the data, we divide it up into
        # chunks and fit each segment. The choice to fit all parameters first is to prevent cases
        # where a chunk is dividing two overlapping points and the variance estimate may be too low.
        for chunk_index, right_break_point in enumerate(chunks):
            left_break_point = chunks[chunk_index-1] if chunk_index != 0 else 0
            # print(chunk_index, left_break_point, right_break_point, chunks)
            segment_x = xdata[left_break_point:right_break_point]
            segment_y = ydata[left_break_point:right_break_point]

            # select the guesses and bounds for this segment
            segment_guess, segment_bounds = [], []
            for guess_index, mean in enumerate(guess[1::step_size]):
                if segment_x[0] < mean:
                    if mean < segment_x[-1]:
                        index_start = guess_index*step_size
                        segment_guess += guess[index_start:index_start+step_size]
                        segment_bounds += bnds[index_start:index_start+step_size]
                    else:
                        break
            if not segment_guess:
                continue

            args = (segment_x, segment_y, baseline_correction)
            opts = {'maxiter': 1000}
            fit_func = bigauss_func if bigauss_fit else gauss_func

            routines = ['SLSQP', 'TNC', 'L-BFGS-B']
            if method:
                routines = [method]

            routine = routines.pop(0)
            if len(bnds) == 0:
                bnds = deepcopy(initial_bounds)
            if baseline_correction:
                jacobian = None
            else:
                jacobian = bigauss_jac if bigauss_fit else gauss_jac
            if debug:
                print('guess and bnds', segment_guess, segment_bounds)
            hessian = None# if bigauss_fit else gauss_hess

            results = [optimize.minimize(fit_func, segment_guess, args, method=routine, bounds=segment_bounds, options=opts, jac=jacobian, hess=hessian)]
            while not results[-1].success and routines:
                routine = routines.pop(0)
                results.append(
                    optimize.minimize(fit_func, segment_guess, args, method=routine, bounds=segment_bounds, options=opts, jac=jacobian))
            if results[-1].success:
                res = results[-1]
            else:
                res = sorted(results, key=attrgetter('fun'))[0]
            n = len(xdata)
            k = len(res.x)
            # this is actually incorrect, but works better...
            # bic = n*np.log(res.fun/n)+k+np.log(n)
            if bigauss_fit:
                bic = 2 * k + 2 * np.log(res.fun / n)
            else:
                bic = res.fun
            res.bic = bic

            for index, value in enumerate(res.x[2::step_size]):
                if value < min_spacing:
                    res.x[2 + index * step_size] = min_spacing
            if bigauss_fit:
                for index, value in enumerate(res.x[3::step_size]):
                    if value < min_spacing:
                        res.x[3 + index * step_size] = min_spacing
            # does this data contain our rt peak?
            res._contains_rt = False
            if rt_peak != 0:
                for i in xrange(1, res.x.size, step_size):
                    mean = res.x[i]
                    lstd = res.x[i + 1]
                    if bigauss_fit:
                        rstd = res.x[i + 2]
                    else:
                        rstd = lstd
                    if mean - lstd * 2 < rt_peak < mean + rstd * 2:
                        res._contains_rt = True

            # TODO: Evaluate the F-test based method
            # if best_res:
            #     cmodel_ssq = best_res.fun
            #     new_model_ssq = res.fun
            #     df = len(xdata)-len(res.x)
            #     f_ratio = (cmodel_ssq-new_model_ssq)/(new_model_ssq/df)
            #     res.p = 1-stats.f.cdf(f_ratio, 1, df)
            #     bic = res.p
            #
            # if not best_res or res.p < best_res.p:
            #     best_res = res
            #     best_fit = np.copy(res.x)
            #     best_rss = res.fun

            fitted_segments[chunk_index].append((bic, res))

    # Figure out the best set of fits
    best_fit = []
    best_rss = 0
    for break_point in sorted(fitted_segments.keys()):
        fits = fitted_segments[break_point]
        lowest_bic = np.inf
        best_segment_res = 0
        best_segment_rss = 0
        for bic, res in fits:
            if bic < lowest_bic or (getattr(best_segment_res, '_contains_rt', False) and res._contains_rt == True):
                if debug:
                    sys.stderr.write('{} < {}'.format(bic, lowest_bic))
                if res._contains_rt == False and best_segment_res != 0 and best_segment_res._contains_rt == True:
                    continue
                best_segment_fit = np.copy(res.x)
                best_segment_res = res
                best_segment_rss = res.fun
                lowest_bic = bic
            if debug:
                sys.stderr.write('{} - best: {}'.format(res, best_segment_fit))
        best_fit += best_segment_fit.tolist()
        best_rss += best_segment_rss

    best_fit = np.array(best_fit)

    if rescale:# and not baseline_correction:
        best_fit[::step_size] *= ydata_original.max()
        if baseline_correction:
            if bigauss_fit:
                best_fit[4::step_size] *= ydata_original.max()
                best_fit[5::step_size] *= ydata_original.max()
            else:
                best_fit[3::step_size] *= ydata_original.max()
                best_fit[4::step_size] *= ydata_original.max()

    return best_fit, best_rss

def findMicro(xdata, ydata, pos, ppm=None, start_mz=None, calc_start_mz=None, isotope=0, spacing=0,
              quant_method='integrate', fragment_scan=False, centroid=False, reporter_mode=False):
    """
    We want to find the boundaries of our isotopic clusters. Basically we search until our gradient
    changes, this assumes it's roughly gaussian and there is little interference
    """
    # find the edges within our tolerance
    tolerance = ppm
    offset = spacing * isotope
    df_empty_index = xdata[ydata == 0]
    if start_mz is None:
        start_mz = xdata[pos]
    fit = True
    if centroid:
        int_val = ydata[pos]
        left, right = pos - 1, pos + 1
        error = get_ppm(start_mz + offset, xdata[pos])
        fit = np.abs(error) < tolerance
        peak = [int_val, xdata[pos], 0]
    else:
        if df_empty_index.size == 0 or not (df_empty_index[0] < xdata[pos] < df_empty_index[-1]):
            left = 0
            right = xdata.size
        else:
            right = np.searchsorted(df_empty_index, xdata[pos])
            left = right - 1
            left, right = (np.searchsorted(xdata, df_empty_index[left], side='left'),
                           np.searchsorted(xdata, df_empty_index[right]))
            right += 1
        new_x = xdata[left:right]
        new_y = ydata[left:right]
        if new_y.sum() == new_y.max():
            peak_mean = new_x[np.where(new_y > 0)][0]
            peaks = (new_y.max(), peak_mean, 0)
            sorted_peaks = [(peaks, get_ppm(start_mz + offset, peak_mean))]
        else:
            peaks, peak_residuals = findAllPeaks(new_x, new_y, min_dist=(new_x[1] - new_x[0]) * 2.0, peak_width_start=1)
            sorted_peaks = sorted(
                [(peaks[i * 3:(i + 1) * 3], get_ppm(start_mz + offset, v)) for i, v in enumerate(peaks[1::3])],
                key=itemgetter(1))
        if (fragment_scan == False or reporter_mode) and not within_tolerance(sorted_peaks, tolerance):
            if calc_start_mz is not None:
                sorted_peaks2 = sorted(
                    [(peaks[i * 3:(i + 1) * 3], get_ppm(calc_start_mz + offset, v)) for i, v in enumerate(peaks[1::3])],
                    key=itemgetter(1))
                if any(filter(lambda x: x[1] < tolerance, sorted_peaks2)):
                    sorted_peaks = sorted_peaks2
                else:
                    fit = False
            else:
                fit = False

        peak = np.array(sorted_peaks[0][0])
        # only go ahead with fitting if we have a stdev. Otherwise, set this to 0
        if peak[2] > 0:
            # peak[0] *= new_y.max()
            int_val = gauss_ndim(new_x, peak).sum()
        else:
            int_val = 0.
        if not fit:
            pass
        error = sorted_peaks[0][1]
    ret_dict = {'int': int_val if fit or (fragment_scan == True and not reporter_mode) else 0., 'int2': int_val, 'bounds': (left, right),
                'params': peak, 'error': error}
    return ret_dict

def within_tolerance(arr, tolerance):
    # arr is a list of tuples with the [1] index for each tuple being the ppm error
    for i in arr:
        if i[1] < tolerance:
            return 1
    return 0

def targeted_search(merged_x, merged_y, x_value, attempts=4, stepsize=3, peak_finding_kwargs=None):
    rt_attempts = 0
    fitting_y = np.copy(merged_y)
    peak_finding_kwargs = peak_finding_kwargs or {}
    debug = peak_finding_kwargs.get('debug')
    found_rt = False
    while rt_attempts < attempts and not found_rt:
        if debug:
            print('MERGED PEAK FINDING ATTEMPT', rt_attempts)
        res, residual = findAllPeaks(
            merged_x,
            fitting_y,
            filter=False,
            bigauss_fit=True,
            rt_peak=x_value,
            **peak_finding_kwargs
        )
        rt_peak = bigauss_ndim(np.array([x_value]), res)[0]
        # we don't do this routine for cases where there are > 5
        found_rt = sum(fitting_y > 0) <= 5 or rt_peak > 0.05
        if not found_rt and rt_peak < 0.05:
            # get the closest peak
            nearest_peak = \
            sorted([(i, np.abs(x_value - i)) for i in res[1::stepsize]], key=itemgetter(1))[0][0]
            # this is tailored to massa spectrometry elution profiles at the moment, and only evaluates for situtations where the rt and peak
            # are no further than a minute apart.
            if np.abs(nearest_peak - x_value) < 1:
                rt_index = find_nearest_index(merged_x, x_value)
                peak_index = find_nearest_index(merged_x, nearest_peak)
                if rt_index < 0:
                    rt_index = 0
                if peak_index == -1:
                    peak_index = len(fitting_y)
                if rt_index != peak_index:
                    grad_len = np.abs(peak_index - rt_index)
                    if grad_len < 4:
                        found_rt = True
                    else:
                        gradient = (np.gradient(fitting_y[rt_index:peak_index]) > 0) if rt_index < peak_index else (
                            np.gradient(fitting_y[peak_index:rt_index]) < 0)
                        if sum(gradient) >= grad_len - 1:
                            found_rt = True
                else:
                    found_rt = True
        if not found_rt:
            if debug:
                print('cannot find rt for', x_value)
                print(merged_x, fitting_y, res, sum(fitting_y > 0))

            fitting_y -= bigauss_ndim(merged_x, res)
            fitting_y[fitting_y < 0] = 0
        rt_attempts += 1

    return (res, residual) if found_rt else (None, np.inf)
