import copy
import six

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