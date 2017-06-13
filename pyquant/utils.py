import copy
import os
import sys
import warnings
from collections import Counter
from functools import partial
from itertools import combinations
from operator import itemgetter

import six
import numpy as np
import pandas as pd
from scipy.misc import comb
from scipy.signal import savgol_filter

from . import PEAK_FINDING_DERIVATIVE, PEAK_FINDING_REL_MAX
from .logger import logger

if six.PY3:
    xrange = range

ERRORS = 'error' if os.environ.get('PYQUANT_DEV', False) == 'True' else 'ignore'

def merge_list(starting_list):
    final_list = []
    for i,v in enumerate(starting_list[:-1]):
        if set(v)&set(starting_list[i+1]):
            starting_list[i+1].extend(list(set(v) - set(starting_list[i+1])))
        else:
            final_list.append(v)
    final_list.append(starting_list[-1])
    return final_list

def findValleys(y, srt):
    peak = y.iloc[srt]
    for left in xrange(srt-1, -1, -1):
        val = y.iloc[left]
        if val == 0 or val > peak:
            break
    right = len(y)
    for right in xrange(srt+1, len(y)):
        val = y.iloc[right]
        if val == 0 or val > peak:
            break
    return left, right

def fit_theo_dist(params, ny, ty):
    right_limit, scaling = params
    index = xrange(len(ny)) if len(ny) > len(ty) else xrange(len(ty))
    exp_dist = pd.Series(0, index=index)
    theo_dist = pd.Series(0, index=index)
    exp_dist += pd.Series(ny)
    exp_dist[int(right_limit):] = 0
    theo_dist += ty
    return ((exp_dist-theo_dist*scaling)**2).sum()

def looper(selected=None, df=None, theo=None, index=0, out=None):
    if out is None:
        out = [0]*len(selected)
    if index != len(selected):
        for i in selected[index]:
            out[index] = i
            for j in looper(selected=selected, df=df, theo=theo, index=index+1, out=out):
                yield j
    else:
        vals = pd.Series([df[i] for i in out])
        vals = (vals/vals.max()).fillna(0)
        residual = ((theo-vals)**2).sum()
        yield (residual, copy.deepcopy(out))


def find_prior_scan(msn_map, current_scan, ms_level=None):
    prior_msn_scans = {}
    for scan_msn, scan_id in msn_map:
        if scan_id == current_scan:
            return prior_msn_scans.get(ms_level if ms_level is not None else scan_msn, None)
        prior_msn_scans[scan_msn] = scan_id
    return None


def find_next_scan(msn_map, current_scan, ms_level=None):
    scan_found = False
    for scan_msn, scan_id in msn_map:
        if scan_found:
            if ms_level is None:
                return scan_id
            elif scan_msn == ms_level:
                return scan_id
        if not scan_found and scan_id == current_scan:
            scan_found = True
    return None


def find_scan(msn_map, current_scan):
    for scan_msn, scan_id in msn_map:
        if scan_id == current_scan:
            return scan_id
    return None


def get_scans_under_peaks(rt_scan_map, found_peaks):
    scans = {}
    for peak_isotope, isotope_peak_data in six.iteritems(found_peaks):
        scans[peak_isotope] = {}
        for xic_peak_index, xic_peak_params in six.iteritems(isotope_peak_data):
            mean, stdl, stdr = xic_peak_params['peak_mean'], xic_peak_params['std'], xic_peak_params['std2']
            left, right = mean - 2 * stdl, mean + 2 * stdr
            scans[peak_isotope][xic_peak_index] = set(
                rt_scan_map[(rt_scan_map.index >= left) & (rt_scan_map.index <= right)].values)
    return scans


def select_window(data, index, size):
    left = index - size
    right = index + size
    if left < 0:
        left = 0
    if right >= len(data) - 1:
        right = -1
    return data[left:] if right == -1 else data[left:right+1]


def find_common_peak_mean(found_peaks, tie_breaker_time=None):
    # We want to find the peak that is common across multiple scans. This is acheived by the following logic.
    # Suppose we have 3 scans providing peaks. To find our reference, it is likely the peak that occurrs across
    # all scans. A problem that we have to face is the x axis is not guaranteed the same between scans, so we need to
    # establish the overlap of the peak and the peak width. This can be imagined as so:
    #
    # Key: | represent peak bounds, x represents mean. WOO ASCII ART!
    #
    # Scan 1:
    #                             |--x--|                |-x-----|             |-------------x---|
    # Scan 2:
    #                         |---x--|        |---x--|
    #
    # Scan 3:
    #                                 |---x--|                         |----x-----|
    #
    # Thus, the left-most peak is likely the peak we are searching for, and we can select the peaks closest to the mean
    # of these 3 scans as the real peak in our experimental data.
    potential_peaks = {}
    # A peak comparsion of A->B is the same as B->A, so we use combinations to do the minimal amount of work
    # First, we remove a level of nesting to make this easier
    new_peaks = {}
    for peaks_label, peaks_info in six.iteritems(found_peaks):
        for ion, peaks in six.iteritems(peaks_info):
            new_peaks[(peaks_label, ion)] = peaks

    for peaks1_key, peaks2_key in combinations(new_peaks.keys(), 2):
        peaks1 = new_peaks[peaks1_key]
        peaks2 = new_peaks[peaks2_key]
        for peak_index1, peak1 in enumerate(peaks1):
            area, mean = peak1.get('total'), peak1.get('mean')
            left_bound, right_bound = mean-peak1.get('std1', 0)*2, mean+peak1.get('std2', peak1.get('std1', 0))*2
            dict_key = (peaks1_key, peak_index1)
            try:
                potential_peaks[dict_key]['intensities'] += area
            except KeyError:
                potential_peaks[dict_key] = {'intensities': area, 'overlaps': set()}
            for peak_index2, peak2 in enumerate(peaks2):
                area2, mean2 = peak2.get('total'), peak2.get('mean')
                left_bound2, right_bound2 = mean2 - peak2.get('std1', 0) * 2, mean2 + peak2.get('std2', peak2.get('std1', 0)) * 2
                # This tests whether there is a overlap.
                if (left_bound <= right_bound2) and (left_bound2 <= right_bound):
                    potential_peaks[dict_key]['overlaps'].add(((peaks2_key, peak_index2), mean2))
                    potential_peaks[dict_key]['intensities'] += area2
    peak_overlaps = [(key, len(overlap_info['overlaps']), overlap_info['intensities']) for key, overlap_info in six.iteritems(potential_peaks)]
    if not peak_overlaps and len(new_peaks) == 1:
        # there is only 1 ion w/ peaks, just pick the biggest peak
        means = [sorted(list(new_peaks.values())[0], key=itemgetter('total'), reverse=True)[0].get('mean')]
    else:
        sorted_peaks = sorted(peak_overlaps, key=itemgetter(1, 2), reverse=True)
        first_likely_peak = sorted_peaks[0]
        most_likely_peaks = [first_likely_peak]
        if len(sorted_peaks) > 1:
            for likely_peak in sorted_peaks[1:]:
                if likely_peak[1:] == first_likely_peak[1:]:
                    most_likely_peaks.append(likely_peak)
                else:
                    break
        # If we have equi-probable peaks, choose the one that is closest to the tie-breaker time, otherwise return the first
        if tie_breaker_time is not None:
            tie_breaker_distances = [((key, peak_index), np.abs(tie_breaker_time - new_peaks[key][peak_index]['mean']))
                                     for (key, peak_index), overlaps, intensity in most_likely_peaks]
            tie_breaker_distances.sort(key=itemgetter(1))
            most_likely_peak = tie_breaker_distances[0][0]
        else:
            most_likely_peak = most_likely_peaks[0][0]

        means = [i[1] for i in potential_peaks[most_likely_peak]['overlaps']]
        # add in the mean of the initial peak
        means.append(new_peaks[most_likely_peak[0]][most_likely_peak[1]].get('mean'))
    return float(sum(means)) / len(means)


def nanmean(arr, empty=0):
    with warnings.catch_warnings():
        warnings.simplefilter(ERRORS, category=RuntimeWarning)
        if any((i for i in arr if not pd.isnull(i))):
            if isinstance(arr, list):
                return np.nanmean(arr)
            elif arr.any():
                return np.nanmean(arr)
        return empty


def divide_peaks(peaks, min_sep=5):
    # We divide up the list of peaks to reduce the number of dimensions each fitting routine is working on
    # to improve convergence speeds
    chunks = argrelextrema(np.abs(peaks), np.less, order=min_sep)[0]
    return chunks


def inffilter(arr):
    with warnings.catch_warnings():
        warnings.simplefilter(ERRORS, category=RuntimeWarning)
        return filter(lambda x: x not in (np.inf, -np.inf), arr)


def naninfmean(arr, empty=0):
    with warnings.catch_warnings():
        warnings.simplefilter(ERRORS, category=RuntimeWarning)
        arr = list(inffilter(arr))
        if any((i for i in arr if not pd.isnull(i))):
            return np.nanmean(arr)
        return empty


def naninfsum(arr, empty=0):
    with warnings.catch_warnings():
        warnings.simplefilter(ERRORS, category=RuntimeWarning)
        arr = list(inffilter(arr))
        if any((i for i in arr if not pd.isnull(i))):
            return np.nansum(arr)
        return empty


ETNS = {
    1: {'C': .0110, 'H': 0.00015, 'N': 0.0037, 'O': 0.00038, 'S': 0.0075},
    2: {'O': 0.0020, 'S': .0421},
    4: {'S': 0.00020}
}

def calculate_theoretical_distribution(peptide=None, elemental_composition=None):
    from pythomics.proteomics.config import RESIDUE_COMPOSITION

    def dio_solve(n, l=None, index=0, out=None):
        if l is None:
            l = [1, 2, 4]
        if out is None:
            out = [0] * len(l)
        if index != len(l):
            for i in xrange(int(n / l[index]) + 1):
                out[index] = i
                for j in dio_solve(n, l=l, index=index + 1, out=out):
                    yield j
        else:
            if n == sum([a * b for a, b in zip(l, out)]):
                yield out

    ETN_P = {}
    if peptide is not None:
        elemental_composition = {}
        aa_counts = Counter(peptide)
        for aa, aa_count in aa_counts.items():
            for element, element_count in RESIDUE_COMPOSITION[aa].items():
                try:
                    elemental_composition[element] += aa_count * element_count
                except KeyError:
                    elemental_composition[element] = aa_count * element_count
        # we lose a water for every peptide bond
        peptide_bonds = len(peptide) - 1
        elemental_composition['H'] -= peptide_bonds * 2
        elemental_composition['O'] -= peptide_bonds
        # and gain a hydrogen for our NH3
        elemental_composition['H'] += 1

    total_atoms = sum(elemental_composition.values())
    for etn, etn_members in ETNS.items():
        p = 0.0
        for isotope, abundance in etn_members.items():
            p += elemental_composition.get(isotope, 0) * abundance / total_atoms
        ETN_P[etn] = p
    tp = 0
    dist = []
    while tp < 0.999:
        p = 0
        for solution in dio_solve(len(dist)):
            p2 = []
            for k, i in zip(solution, [1, 2, 4]):
                petn = ETN_P[i]
                p2.append((comb(total_atoms, k) * (petn ** k)) * ((1 - petn) ** (total_atoms - k)))
            p += np.cumprod(p2)[-1]
        tp += p
        dist.append(p)
    return pd.Series(dist)


def get_classifier():
    import os
    from six.moves.cPickle import load

    pq_dir = os.path.split(__file__)[0]
    handle = open(os.path.join(pq_dir, 'static', 'classifier.pickle'), 'rb')
    classifier = load(handle) if six.PY2 else load(handle, encoding='latin1')

    return classifier

def perform_ml(data, mass_labels):
    import numpy as np
    import sys

    from sklearn.preprocessing import scale
    from scipy.special import logit

    classifier = get_classifier()
    cols = ['Isotopes Found', 'Intensity', 'RT Width', 'Mean Offset', 'Residual', 'R^2', 'SNR']

    for label1 in mass_labels.keys():
        for label2 in mass_labels.keys():
            if label1 == label2:
                continue

            try:
                nd = pd.DataFrame([], columns=[
                    'Label1 Isotopes Found',
                    'Label1 Intensity',
                    'Label1 RT Width',
                    'Label1 Mean Offset',
                    'Label1 Residual',
                    'Label1 R^2',
                    'Label1 SNR',
                    'Label2 Isotopes Found',
                    'Label2 Intensity',
                    'Label2 RT Width',
                    'Label2 Mean Offset',
                    'Label2 Residual',
                    'Label2 R^2',
                    'Label2 SNR',
                ])
                for label, new_label in zip([label1, label2], ['Label1', 'Label2']):
                    for col in cols:
                        nd['{} {}'.format(new_label, col)] = data['{} {}'.format(label, col)]

                nd.replace([-np.inf, np.inf, 'NA'], np.nan, inplace=True)
                non_na_data = nd.dropna().index
                nd.loc[non_na_data, 'Label1 Intensity'] = np.log2(nd.loc[non_na_data, 'Label1 Intensity'].astype(float))
                nd.loc[non_na_data, 'Label2 Intensity'] = np.log2(nd.loc[non_na_data, 'Label2 Intensity'].astype(float))

                nd.replace([-np.inf, np.inf, 'NA'], np.nan, inplace=True)
                non_na_data = nd.dropna().index
                nd.loc[non_na_data, 'Label1 R^2'] = logit(nd.loc[non_na_data, 'Label1 R^2'].astype(float))
                nd.loc[non_na_data, 'Label2 R^2'] = logit(nd.loc[non_na_data, 'Label2 R^2'].astype(float))

                nd.replace([-np.inf, np.inf, 'NA'], np.nan, inplace=True)
                non_na_data = nd.dropna().index
                if not len(non_na_data):
                    continue
                nd.loc[non_na_data, :] = scale(nd.loc[non_na_data, :].values)

                mixed_confidence = '{}/{} Confidence'.format(label1, label2)


                # for a 'good' vs. 'bad' classifier, where 1 is good
                conf_ass = classifier.predict_proba(nd.loc[non_na_data, :])[:, 1] * 10
                data.loc[non_na_data, mixed_confidence] = conf_ass

            except Exception as e:
                import traceback
                sys.stderr.write(
                    'Unable to calculate statistics for {}/{}.\n Traceback: {}'.format(label1, label2, traceback.format_exc()))


def argrelextrema(data, comparator, axis=0, order=1, mode='clip'):
    results = boolrelextrema(data, comparator, axis, order, mode)
    return np.where(results)


def boolrelextrema(data, comparator, axis=0, order=1, mode='clip'):
    """
    A fixed function for scipy's _boolrelextrama that handles data duplication
    :param data:
    :param comparator:
    :param axis:
    :param order:
    :param mode:
    :return:
    """
    if((int(order) != order) or (order < 1)):
        raise ValueError('Order must be an int >= 1')

    datalen = data.shape[axis]
    locs = np.arange(0, datalen)
    data = np.ma.masked_array(data, mask=np.hstack(([1], np.diff(data)))==0)
    if np.ma.is_masked(data):
        locs = locs[np.ma.getmask(data)==False]
        main = data.take(locs, axis=axis, mode=mode)
        results = np.zeros(data.shape, dtype=bool)
        for index, result in enumerate(boolrelextrema(main, comparator, axis=axis, order=order, mode=mode)):
            results[locs[index]] = result
        return results
    else:
        results = np.ones(data.shape, dtype=bool)
        main = data.take(locs, axis=axis, mode=mode)
        for shift in xrange(1, order + 1):
            plus = data.take(locs + shift, axis=axis, mode=mode)
            minus = data.take(locs - shift, axis=axis, mode=mode)
            results &= comparator(main, plus)
            results &= comparator(main, minus)
            if(~results.any()):
                return results
        return results


def merge_peaks(peaks_found, debug=False):
    if len(peaks_found) == 1:
        return peaks_found

    final_peaks = {}
    peak_widths = sorted(peaks_found.keys())
    for peak_width_index in xrange(len(peak_widths) - 1):
        current_width = peak_widths[peak_width_index]
        next_width = peak_widths[peak_width_index + 1]
        current_peaks = peaks_found[current_width]
        next_peaks = peaks_found[next_width]
        smaller_peaks, smaller_minima = current_peaks['peaks'], current_peaks['minima']
        larger_peaks, larger_minima = next_peaks['peaks'], next_peaks['minima']
        if debug:
            sys.stderr.write('{}: {} ---- {}\n'.format(peak_width_index, smaller_peaks, larger_peaks))
        if set(smaller_peaks) == set(larger_peaks) and set(smaller_minima) == set(larger_minima):
            final_peaks[next_width] = next_peaks
            if current_width in final_peaks:
                del final_peaks[current_width]
        else:
            final_peaks[current_width] = peaks_found[current_width]
            if current_width == peak_widths[-2]:
                final_peaks[next_width] = peaks_found[next_width]

    return final_peaks


def find_possible_peaks(xdata, ydata, ydata_peaks, peak_find_method=PEAK_FINDING_REL_MAX, min_dist=None, local_filter_size=0,
                 rt_peak=None, max_peaks=4, peak_width_start=2, snr=0, zscore=0, amplitude_filter=0,
                 peak_width_end=4, fit_negative=False, percentile_filter=0, micro=False, min_slope=None,
                 min_peak_side_width=None, min_peak_width=5, smooth=False):
    PEAK_METHODS = {
        PEAK_FINDING_REL_MAX: partial(
            find_peaks_rel_max,
            peak_width_start=peak_width_start,
            peak_width_end=peak_width_end,
            micro=micro,
        ),
        PEAK_FINDING_DERIVATIVE: partial(
            find_peaks_derivative,
            min_slope=min_slope,
            min_peak_side_width=min_peak_side_width,
            min_peak_width=min_peak_width,
            smooth=smooth,
        ),
    }
    abs_ydata = np.abs(ydata)
    possible_peaks = PEAK_METHODS[peak_find_method](xdata, ydata, ydata_peaks=ydata_peaks)

    for peak_width, peak_info in six.iteritems(possible_peaks):
        row_peaks, minima = peak_info['peaks'], peak_info['minima']
        if min_dist:
            row_peaks = merge_close_peaks(row_peaks, ydata, distance=min_dist)
        if snr or zscore:
            ydata_peaks_std = np.std(ydata_peaks)
            ydata_peaks_median = np.median(ydata_peaks)
            if local_filter_size:
                new_peaks = []
                lost_peaks = {}
                for row_peak in row_peaks:
                    selection = np.abs(select_window(ydata_peaks, row_peak, local_filter_size))
                    ydata_row_peak = np.abs(ydata_peaks[row_peak])
                    local_std = np.std(selection)
                    local_snr = ydata_row_peak / local_std
                    local_zscore = (ydata_row_peak - np.median(selection)) / local_std
                    add_peak = (snr == 0 or local_snr > snr) and \
                               (zscore == 0 or local_zscore >= zscore)
                    if add_peak:
                        new_peaks.append(row_peak)
                    else:
                        lost_peaks[row_peak] = {'snr': local_snr, 'zs': local_zscore}
                logger.debug('{} peaks lost to filtering\n{}\n'.format(len(row_peaks) - len(new_peaks), lost_peaks))
                row_peaks = np.array(new_peaks, dtype=int)
            else:
                if snr and row_peaks.any():
                    logger.debug('{} peaks lost to SNR\n'.format(sum(ydata_peaks[row_peaks] / ydata_peaks_std < snr)))
                if zscore and row_peaks.any():
                    logger.debug('{} peaks lost to zscore\n'.format(
                            sum((ydata_peaks[row_peaks] - ydata_peaks_median) / ydata_peaks_std < zscore)))
                if snr and row_peaks.any():
                    row_peaks = row_peaks[np.abs(ydata_peaks[row_peaks]) / ydata_peaks_std >= snr]
                if zscore and row_peaks.any():
                    row_peaks = row_peaks[
                        (np.abs(ydata_peaks[row_peaks]) - ydata_peaks_median) / ydata_peaks_std >= zscore]

        if amplitude_filter != 0 and row_peaks.any():
            logger.debug(
                '{} peaks lost to amp filter\n{}\n'.format(sum(abs_ydata[row_peaks] < amplitude_filter),
                                                           row_peaks[abs_ydata[row_peaks] < amplitude_filter]))
            row_peaks = row_peaks[np.abs(ydata_peaks[row_peaks]) >= amplitude_filter]

        if percentile_filter and row_peaks.any():
            logger.debug('{} peaks lost to percentile filter\n{}\n'.format(
                sum(np.abs(ydata_peaks[row_peaks]) < np.percentile(abs_ydata, percentile_filter)),
                row_peaks[np.abs(ydata_peaks[row_peaks]) < np.percentile(abs_ydata, percentile_filter)]
            ))
            row_peaks = row_peaks[np.abs(ydata_peaks[row_peaks]) >= np.percentile(abs_ydata, percentile_filter)]
        # Max peaks is to avoid spending a significant amount of time fitting bad data. It can lead to problems
        # if the user is searching the entire ms spectra because of the number of peaks possible to find
        if max_peaks != -1 and row_peaks.size > max_peaks:
            # pick the top n peaks for max_peaks
            if rt_peak:
                # If the user specified a retention time as a guide, select the n peaks closest
                row_peaks = np.sort(row_peaks[np.argsort(np.abs(xdata[row_peaks] - rt_peak)[:max_peaks])])
            else:
                # this selects the row peaks in ydata, reversed the sorting order (to be greatest to least), then
                # takes the number of peaks we allow and then sorts those peaks
                row_peaks = np.sort(row_peaks[np.argsort(
                    np.abs(ydata_peaks[row_peaks]) if fit_negative else ydata_peaks[row_peaks])[::-1]][:max_peaks])

        possible_peaks[peak_width] = {'peaks': row_peaks, 'minima': minima}

    return merge_peaks(possible_peaks)


def find_peaks_rel_max(xdata, ydata, ydata_peaks=None, peak_width_start=2, peak_width_end=4, micro=False):
    ydata_peaks = ydata_peaks if ydata_peaks is not None else ydata
    if peak_width_start > peak_width_end:
        peak_width_end = peak_width_start + 1
    peak_width = peak_width_start
    final_peak = False
    peaks_found = {}
    while peak_width <= peak_width_end or final_peak:
        row_peaks = np.array(argrelextrema(np.abs(ydata_peaks), np.greater, order=peak_width)[0], dtype=int)
        if not row_peaks.size:
            row_peaks = np.array([np.argmax(np.abs(ydata))], dtype=int)
        if len(row_peaks) == 1:
            # We don't need to look for a final peak, we already found a global maximum peak with no other peaks
            final_peak = None
        logger.debug('peak indices: {}\n'.format(row_peaks))

        if ydata_peaks.size:
            minima = np.where(ydata_peaks == 0)[0].tolist()
        else:
            minima = []
        minima.extend(
            [i for i in argrelextrema(np.abs(ydata_peaks), np.less, order=peak_width)[0] if
             i not in minima and i not in row_peaks]
        )
        minima.sort()

        peaks_found[peak_width] = {
            'peaks': row_peaks,
            'minima': minima
        }

        peaks_found[peak_width] = {'peaks': row_peaks, 'minima': minima}
        peak_width += 1
        if peak_width > peak_width_end:
            if final_peak:
                final_peak = False
            elif final_peak is not None and not micro:
                final_peak = True
                peak_width = len(xdata)

    return peaks_found


def get_cross_points(data, pad=True):
    cross_points = []
    signs = np.sign(data)
    for i in xrange(len(signs) - 1):
        if signs[i] != signs[i + 1]:
            cross_points.append(i)

    if cross_points and pad:
        # Pad the ends to deal with the first and last peaks found by prepending the crossing points array
        # with the data point that has the closest opposite sign
        if cross_points[0] != 0:
            cross_points.insert(0, np.argmax(signs[:cross_points[0]] != signs[cross_points[0]]))

        if cross_points[-1] != len(signs)-1:
            not_equal = signs[cross_points[-1] + 1:] != signs[cross_points[-1]+1]
            if not_equal.any():
                cross_points.append(cross_points[-1] + np.argmax(not_equal) + 1)
            else:
                cross_points.append(len(signs)-1)

    return cross_points

def find_peaks_derivative(xdata, ydata, ydata_peaks=None, min_slope=None, rel_peak_height=1.05,
                          min_peak_side_width=2, max_peak_side_width=np.inf,
                          min_peak_width=None, max_peak_width=np.inf, smooth=True):


    # The general strategy here is to identify where the derivative crosses the zero point, and
    # pick out when the sign of the derivative changes in a manner consistent with a peak.
    # A peak using this approach would like look (pw=peak width):
    #
    #          |     left p.w   |  right p.w.  |
    #    ++--+-++++++++++++++++++--------------+++++++++++
    #                           ^ peak max
    #
    #

    peaks_found = {'peaks': [], 'minima': []}
    ydata_peaks = ydata_peaks if ydata_peaks is not None else ydata
    ydata = np.abs(ydata_peaks)

    if min_peak_width is None:
        min_peak_width = int(len(ydata) / 2)
        if min_peak_width > 5:
            min_peak_width = 5

    # We first take the derivative of the data and smooth the derivative to reduce noise
    deriv = np.diff(ydata)
    smoothed_deriv = savgol_smooth(deriv) if smooth else deriv
    cross_points = get_cross_points(smoothed_deriv)
    # Because we take the difference, we need to figure out whether the left or right is the true peak for crosses.
    # By default, cross points returns the left side
    for i in xrange(len(cross_points)):
        index = cross_points[i]
        if index < len(ydata):
            if ydata[index] < ydata[index+1]:
                cross_points[i] = index+1

    if cross_points:
        # Next, we compare crossing points to identify peaks and constrain it using some criteria:
        # slope, peak width
        for i in xrange(1, len(cross_points) - 1):
            left, center, right = cross_points[i - 1], cross_points[i], cross_points[i + 1]
            left_x, center_x, right_x = xdata[left], xdata[center], xdata[right]
            left_y, center_y, right_y = ydata[left], ydata[center], ydata[right]

            logger.debug('Looking at peak %s', (left, center, right, left_x, center_x, right_x, left_y, center_y, right_y))

            # Most basic assumption -- does it look like a peak?
            if not left_y < center_y > right_y:
                logger.debug('%s doesnt look like a peak', center_x)
                continue

            # Does it meet our peak width criteria?
            left_peak_width = center - left
            right_peak_width = right - center
            if not (
                    (
                      (min_peak_side_width <= left_peak_width <= max_peak_side_width) or
                      (min_peak_side_width > left) # This allows us to find peaks that start at the edges of our data
                    ) and (
                      (min_peak_side_width <= right_peak_width <= max_peak_side_width) or
                      ( (len(ydata) - 1) < right + min_peak_side_width) # This allows us to find peaks that start at the edges of our data
                )):
                logger.debug('peak half not wide enough')
                continue

            peak_width = right - left
            if not (min_peak_width < peak_width < max_peak_width):
                logger.debug('peak not wide enough')
                continue

            slope = max(np.abs([
                (center_y - left_y) / (center_x - left_x),
                (center_y - right_y) / (center_x - right_x),
            ]))

            # Both sides of the peak shape must increase beyond a given threshold so we take the minima here
            rel_peak = min(np.abs([
                (center_y / left_y),
                (center_y / right_y),
            ]))

            if min_slope is not None and slope < min_slope:
                logger.debug('not sloped enough')
                continue
            if rel_peak < rel_peak_height:
                logger.debug('not tall enough')
                continue

            peaks_found['peaks'].append(center)
            peaks_found['minima'].append(left)
            peaks_found['minima'].append(right)

    peaks_found['peaks'] = np.array(peaks_found['peaks'])
    peaks_found['minima'] = np.array(peaks_found['minima'])

    # We return our peaks like:
    # { peak_search_width: {'peaks': [], 'minima': []}} to be consistent with other peak finding routines
    return {min_peak_width: peaks_found}


def erode_direction(peaks_found, peak_heights, start, increment, distance):
    peak = peaks_found[start]
    peak_height = peak_heights[start]
    next_index = start + increment
    peaks_to_remove = set([])
    while (next_index >= 0 and next_index < len(peaks_found)) and np.abs(
            peak - peaks_found[next_index]) < distance and peak_height != peak_heights[next_index]:
        peaks_to_remove.add(peaks_found[next_index])
        next_index += increment
    return peaks_to_remove


def merge_close_peaks(peaks_found, ydata, distance=2):
    new_peaks = []
    removed_peaks = set([])
    if peaks_found.any():
        peak_heights = ydata[peaks_found]
        sort_order = np.argsort(peak_heights)[::-1]

        for sort_index, peak_index in enumerate(sort_order):
            peak = peaks_found[peak_index]
            if peak in removed_peaks:
                continue
            new_peaks.append(peak_index)

            removed_peaks |= erode_direction(peaks_found, peak_heights, peak_index, 1, distance)
            removed_peaks |= erode_direction(peaks_found, peak_heights, peak_index, -1, distance)

    return np.array([i for i in peaks_found if i not in removed_peaks])


def estimate_peak_parameters(xdata, ydata, row_peaks, minima_array, fit_negative=False, rel_peak_constraint=0, micro=False,
                             bigauss_fit=False, baseline_correction=False):
    guess = []
    bnds = []
    last_peak = -1
    skip_peaks = set([])
    fitted_peaks = []
    min_spacing = min(np.diff(xdata))/2
    peak_range = xdata[-1] - xdata[0]
    for peak_num, peak_index in enumerate(row_peaks):
        if peak_index in skip_peaks:
            continue
        next_peak = len(xdata) if peak_index == row_peaks[-1] else row_peaks[peak_num + 1]
        fitted_peaks.append(peak_index)
        rel_peak = ydata[peak_index]
        # bounds for fitting the peak mean
        peak_left = xdata[peak_index - 1]
        peak_right = xdata[peak_index + 1] if peak_index + 1 < len(xdata) else xdata[-1]
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
                left = minima_array[left]
            last_peak = peak_index
            right = np.searchsorted(minima_array, peak_index)
            if right >= minima_array.size:
                right = minima_array[-1]
            else:
                try:
                    right = minima_array[right]
                except Exception as e:
                    print(right, ydata, minima_array, peak_index)
            if right > next_peak:
                right = next_peak
            if right < peak_index:
                right = next_peak
            if right >= len(xdata) - 1:
                right = len(xdata) - 1
            bnds.extend([
                (-1.01, rel_peak * rel_peak_constraint) if rel_peak < 0 and fit_negative else (
                rel_peak * rel_peak_constraint, 1.01),
                (peak_left, peak_right) if micro else (xdata[left], xdata[right]),
                (min_spacing, peak_range)
            ])
            if bigauss_fit:
                bnds.extend([(min_spacing, peak_range)])
            peak_values = ydata[left:right]
            peak_indices = xdata[left:right]
        else:
            left = 0
            right = len(xdata) - 1
            bnds.extend([
                (-1.01, rel_peak * rel_peak_constraint) if rel_peak < 0 and fit_negative else (
                rel_peak * rel_peak_constraint, 1.01),
                (peak_left, peak_right) if micro else (xdata[0], xdata[-1]),
                (min_spacing, peak_range)
            ])
            if bigauss_fit:
                bnds.extend([(
                    min_spacing,
                    peak_range)
                ])
            peak_values = ydata[left:right + 1]
            peak_indices = xdata[left:right + 1]

        logger.debug('bounds are {}'.format((peak_index, left, right, peak_values.tolist(), peak_indices.tolist(), bnds)))

        if peak_values.any():
            average = np.average(peak_indices, weights=np.abs(peak_values))
            variance = np.sqrt(np.average((peak_indices - average) ** 2, weights=np.abs(peak_values)))
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
                guess.extend([rel_peak, xdata[peak_index], variance, variance])
            else:
                guess.extend([rel_peak, xdata[peak_index], variance])
            if baseline_correction:
                slope = (ydata[right] - ydata[left]) / (xdata[right] - xdata[left])
                intercept = ((ydata[right] - slope * xdata[right]) + (ydata[left] - slope * xdata[left])) / 2
                guess.extend([slope, intercept])
                bnds.extend([(None, None),
                             (None, None)])

    if not guess:
        average = np.average(xdata, weights=np.abs(ydata))
        variance = np.sqrt(np.average((xdata - average) ** 2, weights=np.abs(ydata)))
        if variance == 0:
            variance = 0.05
        guess = [max(ydata), average, variance]
        if bigauss_fit:
            guess.extend([variance])
        if baseline_correction:
            slope = (ydata[-1] - ydata[0]) / (xdata[-1] - xdata[0])
            intercept = ((ydata[-1] - slope * xdata[-1]) + (ydata[0] - slope * xdata[0])) / 2
            guess.extend([slope, intercept])
            bnds.extend([(None, None),
                         (None, None)])

    if not bnds:
        bnds.extend([(None, None) for i in guess])

    return guess, bnds


def interpolate_data(x, y, gap_limit=2):
    from scipy.interpolate import interp1d
    x, y = np.array(x), np.array(y)
    pos_indices = np.where(y > 0)[0]
    if pos_indices.size >= 2:
        mapper = interp1d(x[pos_indices], y[pos_indices])
        gaps = np.diff(pos_indices)
        to_fill = np.where((gaps > 1) & (gaps <= gap_limit + 1))[0]
        for index in to_fill:
            left = pos_indices[index] + 1
            right = pos_indices[index + 1]
            y[left:right] = mapper(x[left:right])
    return y


def savgol_smooth(ydata, max_window=10):
    window_size = int(len(ydata) / 10)
    if window_size < 5:
        window_size = 5
    elif window_size > max_window:
        window_size = max_window

    if window_size > len(ydata):
        window_size = len(ydata)

    if not window_size % 2:
        window_size -= 1

    return savgol_filter(ydata, window_size, 3)


def get_formatted_mass(mass):
    """
    This helps with looking up masses in a dictionary by mass by ensuring the values 0.3 and 0.30 are represented
    the same
    :param mass:
    :return:
    """
    return float('{0:0.6f}'.format(float(mass)))


def get_scan_resolution(scan):
    """

    :param scan: A scan series (pandas series)
    :return:
    """
    from .peaks import findAllPeaks

    max_peak = np.argmax(scan.values)

    left = max_peak-30
    right = max_peak+30
    if left < 0:
        left = 0
    if right > len(scan):
        right = len(scan)

    x = scan.index.values[left:right].astype(float)
    y = scan.values[left:right].astype(float)
    peaks, residual = findAllPeaks(x, y, peak_width_start=1, max_peaks=-1)
    if peaks is None or not peaks.any():
        print('no peaks in ', scan)
        return 0

    largest_peak = np.argmax(peaks[0::3])

    fwhm = peaks[2+largest_peak*3]*2.355
    return peaks[1+largest_peak*3]/fwhm
