import os
import sys
from collections import defaultdict, OrderedDict
from copy import deepcopy
from operator import itemgetter, attrgetter

import numpy as np
import six
from pythomics.proteomics.config import CARBON_NEUTRON
from scipy import optimize
from scipy.signal import convolve, kaiser

if os.environ.get('PYQUANT_DEV', False) == 'True':
    try:
        import pyximport; pyximport.install(setup_args={'include_dirs': np.get_include()}, reload_support=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        pass

from pyquant.cpeaks import bigauss_func, gauss_func, bigauss_ndim, gauss_ndim, bigauss_jac,\
    gauss_jac, find_nearest, find_nearest_index, get_ppm
from . import PEAK_FINDING_REL_MAX, PEAK_FIT_MODE_AVERAGE, PEAK_FIT_MODE_FAST, PEAK_FIT_MODE_SLOW
from .logger import logger
from .utils import (
    divide_peaks,
    find_possible_peaks,
    estimate_peak_parameters,
    interpolate_data,
    savgol_smooth,
    subtract_baseline,
)

if six.PY3:
    xrange = range

_epsilon = np.sqrt(np.finfo(float).eps)

def findEnvelope(xdata, ydata, measured_mz=None, theo_mz=None, max_mz=None, precursor_ppm=5, isotope_ppm=2.5,
                 isotope_ppms=None, charge=2, debug=False, isotope_offset=0, isotopologue_limit=-1,
                 theo_dist=None, label=None, skip_isotopes=None, last_precursor=None, quant_method='integrate',
                 reporter_mode=False, fragment_scan=False, centroid=False, contaminant_search=True):
    # returns the envelope of isotopic peaks as well as micro envelopes  of each individual cluster
    spacing = CARBON_NEUTRON / float(charge)
    start_mz = measured_mz if isotope_offset == 0 else measured_mz + isotope_offset * CARBON_NEUTRON / float(charge)
    initial_mz = start_mz
    if max_mz is not None:
        max_mz = max_mz - spacing * 0.9 if isotope_offset == 0 else max_mz + isotope_offset * CARBON_NEUTRON * 0.9 / float(charge)
    if isotope_ppms is None:
        isotope_ppms = {}
    tolerance = isotope_ppms.get(0, precursor_ppm) / 1000000.0
    env_dict, micro_dict, ppm_dict = OrderedDict(), OrderedDict(), OrderedDict()
    empty_dict = {
        'envelope': env_dict,
        'micro_envelopes': micro_dict,
        'ppms': ppm_dict
    }

    non_empty = xdata[ydata > 0]
    if len(non_empty) == 0:
        if debug:
            print('data is empty')
        return empty_dict
    first_mz = find_nearest(non_empty, start_mz)

    isotope_index = 0
    use_theo = False
    # This is purposefully verbose to be more explicit
    if reporter_mode == False and fragment_scan == False:
        while get_ppm(start_mz, first_mz) > tolerance:
            # let's try using our theoretical mass
            first_mz = find_nearest(non_empty, theo_mz)
            if get_ppm(theo_mz, first_mz) > tolerance:
                # let's check our last boundary. This allows for drift in m/z values
                # as scans progress instead of enforcing the m/z at the first
                # observed instance of a given m/z
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
    start_info = findMicro(
        xdata,
        ydata,
        start_index,
        ppm=tolerance,
        start_mz=start_mz,
        calc_start_mz=theo_mz,
        quant_method=quant_method,
        reporter_mode=reporter_mode,
        fragment_scan=fragment_scan,
        centroid=centroid
    )
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
                closest_contaminant = find_nearest(non_empty, start - CARBON_NEUTRON / float(i))
                closest_contaminant_index = find_nearest_index(xdata, closest_contaminant)
                contaminant_bounds = findMicro(xdata, ydata, closest_contaminant_index, ppm=precursor_tolerance,
                                               calc_start_mz=start, start_mz=start, isotope=-1,
                                               spacing=CARBON_NEUTRON / float(i),
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

    return {
        'envelope': env_dict,
        'micro_envelopes': micro_dict,
        'ppms': ppm_dict,
    }

def findAllPeaks(xdata, ydata_original, min_dist=0, method=None, local_filter_size=0, filter=False, peak_boost=False, bigauss_fit=False,
                 rt_peak=None, mrm=False, max_peaks=4, debug=False, peak_width_start=3, snr=0, zscore=0, amplitude_filter=0,
                 peak_width_end=4, fit_baseline=False, rescale=True, fit_negative=False, percentile_filter=0, micro=False,
                 method_opts=None, smooth=False, r2_cutoff=None, peak_find_method=PEAK_FINDING_REL_MAX, min_slope=None,
                 min_peak_side_width=3, gap_interpolation=0, min_peak_width=None, min_peak_increase=None, chunk_factor=0.1,
                 fit_mode=PEAK_FIT_MODE_AVERAGE, baseline_subtraction=False, **kwargs):

    # Deprecation things
    if 'baseline_correction' in kwargs:
        fit_baseline = kwargs['baseline_correction']

    if micro:
        fit_baseline = False

    if not micro and gap_interpolation:
        ydata_original = interpolate_data(xdata, ydata_original, gap_limit=gap_interpolation)

    rel_peak_constraint = (0.0 if fit_baseline else 0.5)
    original_max = np.abs(ydata_original).max() if fit_negative else ydata_original.max()
    amplitude_filter /= original_max
    ydata = ydata_original / original_max
    ydata_peaks = np.copy(ydata)

    if baseline_subtraction:
        ydata_peaks = subtract_baseline(ydata_peaks)

    if smooth and len(ydata) > 5:
        ydata_peaks = savgol_smooth(ydata_peaks)


    if filter or peak_boost:
        if len(ydata) >= 5:
            ydata_peaks = convolve(ydata_peaks, kaiser(10, 12), mode='same')

    ydata_peaks[np.isnan(ydata_peaks)] = 0

    ydata_peaks /= (np.abs(ydata_peaks).max() if fit_negative else ydata_peaks.max())

    final_peaks = find_possible_peaks(
        xdata,
        ydata,
        ydata_peaks,
        peak_find_method=peak_find_method,
        peak_width_start=peak_width_start,
        peak_width_end=peak_width_end,
        snr=snr,
        zscore=zscore,
        rt_peak=rt_peak,
        amplitude_filter=amplitude_filter,
        fit_negative=fit_negative,
        percentile_filter=percentile_filter,
        local_filter_size=local_filter_size,
        micro=micro,
        min_slope=min_slope,
        min_dist=min_dist,
        min_peak_side_width=min_peak_side_width,
        min_peak_width=min_peak_width,
        smooth=smooth,
        min_peak_increase=min_peak_increase,
    )

    # Next, for fitting multiple peaks, we want to divide up the space so we are not fitting peaks that
    # have no chance of actually impacting one another.
    CHUNK_MAP = {
        PEAK_FIT_MODE_SLOW: 0.1,
        PEAK_FIT_MODE_AVERAGE: 0.5,
        PEAK_FIT_MODE_FAST: 1.0,
    }
    chunks = divide_peaks(
        np.abs(ydata_peaks),
        min_sep=5 if 5 > peak_width_end else peak_width_end,
        chunk_factor=CHUNK_MAP[fit_mode]
    )
    if not chunks.any() or chunks[-1] != len(ydata_peaks):
        chunks = np.hstack((chunks, len(ydata_peaks)))

    logger.debug('found: {}\n'.format(final_peaks))

    step_size = 4 if bigauss_fit else 3
    if fit_baseline:
        step_size += 2
    min_spacing = min(np.diff(xdata)) / 2
    peak_range = xdata[-1] - xdata[0]
    # initial bound setup
    initial_bounds = [(-1.01, 1.01), (xdata[0], xdata[-1]), (min_spacing, peak_range)]
    if bigauss_fit:
        initial_bounds.extend([(min_spacing, peak_range)])
    if fit_baseline:
        initial_bounds.extend([(None, None), (None, None)])
        # print(final_peaks)
    if debug:
        sys.stderr.write('final peaks: {}\n'.format(final_peaks))

    fitted_segments = defaultdict(list)
    for peak_width, peak_info in final_peaks.items():
        row_peaks = peak_info['peaks']
        if not row_peaks.any():
            continue

        minima_array = np.array(peak_info['minima'], dtype=np.long)
        # Now that we have estimated the parameters for fitting all the data, we divide it up into
        # chunks and fit each segment. The choice to fit all parameters first is to prevent cases
        # where a chunk is dividing two overlapping points and the variance estimate may be too low.
        for chunk_index, right_break_point in enumerate(chunks):
            left_break_point = chunks[chunk_index - 1] if chunk_index != 0 else 0
            segment_x = xdata[left_break_point:right_break_point]
            segment_y = deepcopy(ydata[left_break_point:right_break_point])

            if not segment_y.any():
                continue

            if not micro:
                segment_max = np.abs(segment_y).max()
                segment_y /= segment_max

            segment_row_peaks = []
            segment_minima_array = []
            for i in row_peaks:
                if i >= left_break_point:
                    if i < right_break_point:
                        segment_row_peaks.append(i-left_break_point)
                    else:
                        break

            for i in minima_array:
                if i >= left_break_point:
                    if i < right_break_point:
                        segment_minima_array.append(i-left_break_point)
                    else:
                        break

            if not segment_row_peaks:
                continue

            # Get peak parameter estimates and boundaries for this segment
            segment_guess, segment_bounds = estimate_peak_parameters(
                segment_x,
                segment_y,
                np.array(segment_row_peaks),
                np.array(segment_minima_array),
                fit_negative=fit_negative,
                rel_peak_constraint=rel_peak_constraint,
                micro=micro,
                bigauss_fit=bigauss_fit,
                fit_baseline=fit_baseline
            )

            if not segment_guess:
                continue

            args = (segment_x, segment_y, fit_baseline)
            opts = method_opts or {'maxiter': 1000}

            # Because the amplitude of peaks can vary wildly, we have to make sure our tolerance matters for the
            # smallest peaks. i.e. if we are fitting two peaks, one with an amplitude of 20M and another with 10000,
            # changes in the smaller peak will be below our tolerance and the minimization routine can ignore them

            if 'ftol' not in opts and not micro:
                min_tol = 1e-10
                for i, j in zip(segment_bounds, segment_guess):
                    abs_i = np.abs(i[0]) if i[0] else None
                    abs_j = np.abs(j) if j else None
                    if abs_i and abs_i < min_tol:
                        min_tol = abs_i/5.
                    if abs_j and abs_j < min_tol:
                        min_tol = abs_j/5.
                opts['ftol'] = min_tol

            fit_func = bigauss_func if bigauss_fit else gauss_func

            routines = ['SLSQP', 'TNC', 'L-BFGS-B']
            if method:
                routines = [method]

            routine = routines.pop(0)
            if len(segment_bounds) == 0:
                segment_bounds = deepcopy(initial_bounds)

            # Check that the bounds for the mean are within the segment so the optimizer doesn't try and cheat
            # by going to solutions outside of the data
            if not micro:
                for i in xrange(1, len(segment_bounds), step_size):
                    if segment_bounds[i][0] < segment_x[0]:
                        segment_bounds[i] = (segment_x[0], segment_bounds[i][1])
                    if segment_bounds[i][1] > segment_x[-1]:
                        segment_bounds[i] = (segment_bounds[i][0], segment_x[-1])

            if fit_baseline:
                jacobian = None
            else:
                jacobian = bigauss_jac if bigauss_fit else gauss_jac

            if debug:
                print('left and right segments', xdata[left_break_point], xdata[right_break_point-1])
                print('guess and bnds', segment_guess, segment_bounds)
            hessian = None  # if bigauss_fit else gauss_hess

            results = [optimize.minimize(fit_func, segment_guess, args, method=routine, bounds=segment_bounds, options=opts,
                                         jac=jacobian, hess=hessian, tol=1e-3)]
            while not results[-1].success and routines:
                routine = routines.pop(0)
                results.append(
                    optimize.minimize(fit_func, segment_guess, args, method=routine, bounds=segment_bounds, options=opts,
                                      jac=jacobian))
            if results[-1].success:
                res = results[-1]
            else:
                res = sorted(results, key=attrgetter('fun'))[0]
            n = len(segment_x)
            k = len(res.x)
            bic = n * np.log(res.fun / n) + k * np.log(n)
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
                    if rt_peak is not None and mean - lstd * 2 < rt_peak < mean + rstd * 2:
                        res._contains_rt = True

            if not micro:
                # Rescale our data back
                # Amplitude
                res.x[::step_size] *= segment_max
                if fit_baseline:
                    # Slope
                    res.x[step_size-2::step_size] *= segment_max
                    # Intercept
                    res.x[step_size-1::step_size] *= segment_max

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

            fitted_segments[(peak_width, chunk_index)].append((bic, res))

    # Figure out the best set of fits
    segment_order = sorted(fitted_segments.keys(), key=itemgetter(0, 1))
    best_fits = {peak_width: {
        'fit': [],
        'residual': 0,
    } for (peak_width, chunk_index) in segment_order}
    for key in segment_order:
        peak_width = key[0]
        fits = fitted_segments[key]
        lowest_bic = np.inf
        best_segment_res = 0
        best_segment_fit = None
        for bic, res in fits:
            if bic < lowest_bic or (getattr(best_segment_res, '_contains_rt', False) != True and res._contains_rt == True):
                if debug:
                    sys.stderr.write('{} < {}\n'.format(bic, lowest_bic))
                if res._contains_rt == False and best_segment_res != 0 and best_segment_res._contains_rt == True:
                    continue
                if debug:
                    print('NEW BEST!', res, 'old was', best_segment_res)
                best_segment_fit = np.copy(res.x)
                best_segment_res = res
                lowest_bic = bic
            if debug:
                sys.stderr.write('{} - best: {}\n'.format(res, best_segment_fit))
        else:
            if best_segment_fit is None:
                return np.array([]), np.inf
            best_fits[peak_width]['fit'] += best_segment_fit.tolist()
            best_fits[peak_width]['residual'] += lowest_bic
            best_fits[peak_width]['contains_rt'] = not best_segment_res._contains_rt  # this is so it sorts lower

    best_fit = sorted(
        ((best_fits[key[0]]['contains_rt'], best_fits[key[0]]['residual'], best_fits[key[0]]['fit']) for key in
         segment_order),
        key=itemgetter(0, 1)
    )

    if not best_fit:
        return (np.array([]), np.inf)

    best_fit = np.array(best_fit[0][2])

    # If the user only wants a certain number of peaks, enforce that now
    if max_peaks != -1:
        # pick the top n peaks for max_peaks
        if rt_peak:
            means = best_fit[1::step_size]
            # If the user specified a retention time as a guide, select the n peaks closest
            peak_indices = np.argsort(np.abs(means - rt_peak))[:max_peaks]
        else:
            # Return the top n highest peaks
            amplitudes = best_fit[0::step_size]
            peak_indices = sorted(np.argsort(amplitudes)[::-1][:max_peaks])
        best_fit = np.hstack((best_fit[i*step_size:(i+1)*step_size] for i in peak_indices))

    peak_func = bigauss_ndim if bigauss_fit else gauss_ndim
    # Get rid of peaks with low r^2
    if not micro and r2_cutoff is not None:
        final_fit = np.array([])
        for peak_index in xrange(0, len(best_fit), step_size):

            peak_info = best_fit[peak_index:peak_index + step_size]
            amplitude, mean, std = peak_info[:3]
            left = mean - 2 * std
            right = mean + 2 * peak_info[3] if bigauss_fit else mean + 2 * std

            # Establish a goodness of fit using the coefficient of determination (the r^2) value for each peak.
            # Because the input data can have multiple peaks, we calculate a r^2 that considers the variance around this peak.
            curve_indices = (xdata >= left) & (xdata <= right)
            fitted_data = ydata[curve_indices]
            fitted_x = xdata[curve_indices]
            for other_peak_index in xrange(0, len(best_fit), step_size):
                if other_peak_index == peak_index:
                    continue
                fitted_data -= peak_func(fitted_x, best_fit[other_peak_index:other_peak_index + step_size])
            ss_tot = np.sum((fitted_data - np.mean(fitted_data)) ** 2)
            explained_data = peak_func(fitted_x, peak_info)
            if fit_baseline:
                explained_data += fitted_x*peak_info[-2]+peak_info[-1]
            ss_res = np.sum((fitted_data - explained_data) ** 2)
            coeff_det = 1 - (ss_res / ss_tot)
            if coeff_det >= r2_cutoff:
                final_fit = np.hstack((final_fit, peak_info))

        best_fit = final_fit

    residual = sum((ydata-peak_func(xdata, best_fit))**2)

    if rescale:  # and not fit_baseline:
        best_fit[::step_size] *= original_max
        if fit_baseline:
            # Slope
            best_fit[step_size-2::step_size] *= original_max
            # Intercept
            best_fit[step_size-1::step_size] *= original_max

    return best_fit, residual


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
            peaks, peak_residuals = findAllPeaks(new_x, new_y, min_dist=(new_x[1] - new_x[0]) * 2.0, peak_width_start=1, micro=True)
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

def targeted_search(merged_x, merged_y, x_value, attempts=4, max_peak_distance=1., peak_finding_kwargs=None):
    rt_attempts = 0
    fitting_y = np.copy(merged_y)
    find_peaks_kwargs = {
        'filter': False,
        'bigauss_fit': True,
        'rt_peak': x_value,
    }
    peak_finding_kwargs = peak_finding_kwargs or {}
    if peak_finding_kwargs:
        assert isinstance(peak_finding_kwargs, dict), 'peak_finding_kwargs must be a dictionary'
        find_peaks_kwargs.update(peak_finding_kwargs)
    debug = peak_finding_kwargs.get('debug')
    found_rt = False
    stepsize = 3
    if find_peaks_kwargs.get('bigauss_fit'):
        stepsize += 1
    if find_peaks_kwargs.get('fit_baseline'):
        stepsize += 2
    while rt_attempts < attempts and not found_rt:
        logger.debug('MERGED PEAK FINDING ATTEMPT %s', rt_attempts)
        res, residual = findAllPeaks(
            merged_x,
            fitting_y,
            **find_peaks_kwargs
        )
        if not res.any():
            return (None, np.inf)
        rt_peak = bigauss_ndim(np.array([x_value]), res)[0]
        # we don't do this routine for cases where there are > 5
        found_rt = sum(fitting_y > 0) <= 5 or rt_peak > 0.05
        if not found_rt and rt_peak < 0.05:
            # get the closest peak
            nearest_peak = \
            sorted([(i, np.abs(x_value - i)) for i in res[1::stepsize]], key=itemgetter(1))[0][0]
            # this is tailored to mass spectrometry elution profiles at the moment, and only evaluates for situtations where the rt and peak
            # are no further than a minute apart.
            if np.abs(nearest_peak - x_value) < max_peak_distance:
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
