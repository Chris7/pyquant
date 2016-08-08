import copy
import six
import warnings
from itertools import combinations
from operator import itemgetter

import numpy as np
import pandas as pd

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
            mean, stdl, stdr = xic_peak_params['mean'], xic_peak_params['std'], xic_peak_params['std2']
            left, right = mean - 2 * stdl, mean + 2 * stdr
            scans[peak_isotope][xic_peak_index] = set(
                rt_scan_map[(rt_scan_map.index >= left) & (rt_scan_map.index <= right)].values)
    return scans


def select_window(x, index, size):
    left = index - size
    right = index + size
    if left < 0:
        left = 0
    if right >= len(x) - 1:
        right = -1
    return x[left:] if right == -1 else x[left:right+1]


def find_common_peak_mean(found_peaks):
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
        means = [sorted(new_peaks.values()[0], key=itemgetter('total'), reverse=True)[0].get('mean')]
    else:
        most_likely_peak = sorted(peak_overlaps, key=itemgetter(1, 2), reverse=True)[0]
        means = [i[1] for i in potential_peaks[most_likely_peak[0]]['overlaps']]
        # add in the mean of the initial peak
        means.append(new_peaks[most_likely_peak[0][0]][most_likely_peak[0][1]].get('mean'))
    return float(sum(means)) / len(means)


def nanmean(arr):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(arr)


def divide_peaks(peaks):
    # We divide up the list of peaks to reduce the number of dimensions each fitting routine is working on
    # to improve convergence speeds
    from scipy.signal import argrelmin
    chunks = argrelmin(peaks, order=5)[0]
    return chunks