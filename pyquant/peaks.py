import os
from collections import Counter
from scipy.misc import comb
import pandas as pd
import six
from pythomics.proteomics.config import RESIDUE_COMPOSITION
import numpy as np
if os.environ.get('PYQUANT_DEV', False) == 'True':
    try:
        import pyximport; pyximport.install(setup_args={'include_dirs': np.get_include()}, reload_support=True)
    except:
        pass
from .cpeaks import *

if six.PY3:
    xrange = range

ETNS = {1: {'C': .0110, 'H': 0.00015, 'N': 0.0037, 'O': 0.00038, 'S': 0.0075},
        2: {'O': 0.0020, 'S': .0421},
        4: {'S': 0.00020}}

_epsilon = np.sqrt(np.finfo(float).eps)

def calculate_theoretical_distribution(peptide):
    def dio_solve(n, l=None, index=0, out=None):
        if l is None:
            l = [1,2,4]
        if out is None:
            out = [0]*len(l)
        if index != len(l):
            for i in xrange(int(n/l[index])+1):
                out[index] = i
                for j in dio_solve(n, l=l, index=index+1, out=out):
                    yield j
        else:
            if n == sum([a*b for a,b in zip(l,out)]):
                yield out

    ETN_P = {}
    element_composition = {}
    aa_counts = Counter(peptide)
    for aa, aa_count in aa_counts.items():
        for element, element_count in RESIDUE_COMPOSITION[aa].items():
            try:
                element_composition[element] += aa_count*element_count
            except KeyError:
                element_composition[element] = aa_count*element_count
    # we lose a water for every peptide bond
    peptide_bonds = len(peptide)-1
    element_composition['H'] -= peptide_bonds*2
    element_composition['O'] -= peptide_bonds
    # and gain a hydrogen for our NH3
    element_composition['H'] += 1

    total_atoms = sum(element_composition.values())
    for etn, etn_members in ETNS.items():
        p = 0.0
        for isotope, abundance in etn_members.items():
            p += element_composition.get(isotope, 0)*abundance/total_atoms
        ETN_P[etn] = p
    tp = 0
    dist = []
    while tp < 0.999:
        p=0
        for solution in dio_solve(len(dist)):
            p2 = []
            for k, i in zip(solution, [1, 2, 4]):
                petn = ETN_P[i]
                p2.append((comb(total_atoms, k)*(petn**k))*((1-petn)**(total_atoms-k)))
            p+=np.cumprod(p2)[-1]
        tp += p
        dist.append(p)
    return pd.Series(dist)


def buildEnvelope(peaks_found=None, isotopes=None, gradient=False, rt_window=None, start_rt=None, silac_label=None):
    isotope_index = {}
    df = pd.DataFrame(columns=rt_window if rt_window is not None else [])
    for silac_isotope, isotope_data in peaks_found.iteritems():
        if isotopes and silac_isotope not in isotopes:
            continue
        temp_data = []
        for envelope, micro_envelope in zip(
                isotope_data.get('envelope', []),
                isotope_data.get('micro_envelopes', [])):
            if envelope:
                iso_df = envelope['df']
                rt = iso_df.name

            if micro_envelope and micro_envelope.get('info'):
                start, end = micro_envelope['info'][0], micro_envelope['info'][1]
                selector = iso_df.index[range(start,end+1)]
                temp_series = pd.Series(iso_df[selector], index=selector)
                try:
                    series_index = isotope_index[silac_isotope]
                except KeyError:
                    series_index = temp_series.idxmax()
                    isotope_index[silac_isotope] = series_index
                temp_int = integrate.simps(temp_series.values)
                temp_mean = np.average(temp_series.index.values, weights=temp_series.values)
                temp_data.append((temp_mean, {'int': temp_int, 'index': series_index, 'rt': rt}))

        exclude = {}
        data = [i[0] for i in temp_data]
        if len(data) > 4:
            data_median, data_std = np.median(data), np.std(data)
            cutoff = 2*data_std
            cut_fun = lambda i: abs(i-data_median)>cutoff
            exclude = set(filter(cut_fun, data))
        for i in temp_data:
            if i[0] in exclude:
                continue
            temp_int = i[1]['int']
            series_index = i[1]['index']
            rt = i[1]['rt']
            if series_index in df.index:
                if rt in df and not pd.isnull(df.loc[series_index, rt]):
                    df.loc[series_index, rt] += temp_int
                else:
                    df.loc[series_index, rt] = temp_int
            else:
                df = df.append(pd.DataFrame(temp_int, columns=[rt], index=[series_index]))

    # THis is experimental
    if not df.empty:
        delta_df = pd.rolling_apply(df, 2, lambda x: x[1]/x[0]).fillna(0)
        ndfs = []
        indices = df.index
        for index, row in delta_df.T.iterrows():
            exclude = False
            for i,v in enumerate(row):
                if v > 3:
                    exclude = True
                    break
            ndfs.append(df.loc[indices[:i] if exclude else indices,index])
        df = pd.concat(ndfs, axis=1) if ndfs else df

        ndfs = []
        from scipy.signal import argrelmax
        closest = sorted([(i, abs(float(v)-start_rt)) for i,v in enumerate(df.columns)], key=itemgetter(1))[0][0]
        isotope_coms = []
        for index, values in df.iterrows():
            y = values.fillna(0)
            maximas = y.iloc[argrelmax(y.fillna(0).values)[0]]
            if len(maximas) == 0:
                ndfs.append(pd.DataFrame(y))
                continue
            # check for peaks outside our window
            # neighboring_peaks = filter(lambda x: abs(x)>0.3, maximas.index-start_rt)
            # if not neighboring_peaks and len(maximas) <= 1:
            #     ndfs.append(pd.DataFrame(y))
            #     continue
            peaks = sorted([(index, abs(float(index)-start_rt), v) for index, v in maximas.iteritems() if v], key=itemgetter(1))
            # choose the peak closest to our retention time
            # if any([i for i in peaks if i[1]<0.30]):
            #     peaks.sort(key=itemgetter(2), reverse=True)
            #     srt = y.index.searchsorted(peaks[0][0])
            # else:
            srt = find_nearest(y.index.values, peaks[0][0])
            # find the left/right from our peak
            # see if we're increasing to the left, otherwise assume we are to the right
            left, right = findPeak(y, srt)
            peak_data = y.iloc[left:right]
            peak_com = np.average(peak_data.index, weights=peak_data.values)
            isotope_coms.append(peak_com)
            ndfs.append(pd.DataFrame(y.iloc[left:right]))
        df = pd.concat(ndfs, axis=1).T if ndfs else df
    # THis is experimental

    rt_data = df.fillna(0).sum(axis=0).sort_index()
    rt_data.index = rt_data.index.astype(float)
    int_val = integrate.trapz(rt_data.values, rt_data.index.values) if not rt_data.empty else 0
    return {'data': df, 'integration': int_val, 'rt': rt_data}
