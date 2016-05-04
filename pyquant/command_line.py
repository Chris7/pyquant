from __future__ import division, unicode_literals, print_function
import sys
from string import Template
import gzip
import signal
import base64
import json
import os
import copy
import operator
import traceback
import pandas as pd
import numpy as np
import random
import six

if six.PY3:
    xrange = range

from itertools import groupby, combinations
from collections import OrderedDict, defaultdict
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from multiprocessing import Process, Queue
from six.moves.queue import Empty

try:
    from profilestats import profile
    from memory_profiler import profile as memory_profiler
except ImportError:
    pass

from datetime import datetime, timedelta
from scipy import integrate
from scipy.stats import linregress
from scipy.interpolate import UnivariateSpline
from scipy.ndimage.filters import gaussian_filter1d

from pythomics.proteomics.parsers import GuessIterator
from pythomics.proteomics import config
from . import peaks

# def line_profiler(view=None, extra_view=None):
#     import line_profiler
#
#     def wrapper(view):
#         def wrapped(*args, **kwargs):
#             prof = line_profiler.LineProfiler()
#             prof.add_function(view)
#             if extra_view:
#                 [prof.add_function(v) for v in extra_view]
#             with prof:
#                 resp = view(*args, **kwargs)
#             prof.print_stats()
#             return resp
#         return wrapped
#     if view:
#         return wrapper(view)
#     return wrapper

description = """
PyQuant is a quantification program for mass spectrometry data. It attempts to be a general implementation to quantify
an assortment of datatypes and allows a high degree of customization for how data is to be quantified.
"""

ION_CUTOFF = 2

CRASH_SIGNALS = {signal.SIGSEGV, }

class Reader(Process):
    def __init__(self, incoming, outgoing, raw_file=None, spline=None, rt_window=None):
        super(Reader, self).__init__()
        self.incoming = incoming
        self.outgoing = outgoing
        self.scan_dict = {}
        self.access_times = {}
        self.raw_path = raw_file
        self.spline = spline
        self.rt_window = rt_window

    def run(self):
        raw = GuessIterator(self.raw_path, full=True, store=False)
        for scan_request in iter(self.incoming.get, None):
            thread, scan_id, mz_start, mz_end = scan_request
            d = self.scan_dict.get(scan_id)
            if not d:
                scan = raw.getScan(scan_id)
                if scan is not None:
                    rt = scan.rt
                    if self.rt_window is not None and not any([(i[0] < float(rt) < i[1]) for i in self.rt_window]):
                        d = None
                    else:
                        scan_vals = np.array(scan.scans)
                        if self.spline:
                            scan_vals[:,0] = scan_vals[:,0]/(1-self.spline(scan_vals[:,0])/1e6)
                        # add to our database
                        d = {
                            'vals': scan_vals,
                            'rt': scan.rt,
                            'title': scan.title,
                            'mass': scan.mass,
                            'charge': scan.charge,
                            'centroid': getattr(scan, 'centroid', False)
                        }
                        self.scan_dict[scan_id] = d
                        # the scan has been stored, delete it
                    del scan
                else:
                    d = None
            if d is not None and (mz_start is not None or mz_end is not None):
                out = copy.deepcopy(d)
                mz_start = 0 if mz_start is None else mz_start
                mz_end = out['vals'][-1,0]+1 if mz_end is None else mz_end
                out['vals'] = out['vals'][np.where((out['vals'][:,0]>=mz_start) & (out['vals'][:,0]<=mz_end))]
                self.outgoing[thread].put(out)
            else:
                self.outgoing[thread].put(d)
            now = datetime.now()
            self.access_times[scan_id] = now
            # evict scans we have not accessed in over 5 minutes
            cutoff = now-timedelta(minutes=5)
            to_delete = []
            for i,v in self.access_times.items():
                if v < cutoff:
                    del self.scan_dict[i]
                    to_delete.append(i)
            for i in sorted(to_delete, reverse=True):
                del self.access_times[i]
        sys.stderr.write('reader done\n')


class Worker(Process):
    def __init__(self, queue=None, results=None, precision=6, raw_name=None, mass_labels=None, isotope_ppms=None,
                 debug=False, html=False, mono=False, precursor_ppm=5.0, isotope_ppm=2.5, quant_method='integrate',
                 reader_in=None, reader_out=None, thread=None, fitting_run=False, msn_rt_map=None, reporter_mode=False,
                 spline=None, isotopologue_limit=-1, labels_needed=1, overlapping_mz=False, min_resolution=0, min_scans=3,
                 quant_msn_map=None, mrm=False, mrm_pair_info=None, peak_cutoff=0.05, ratio_cutoff=1, replicate=False,
                 ref_label=None, max_peaks=4, parser_args=None):
        super(Worker, self).__init__()
        self.precision = precision
        self.precursor_ppm = precursor_ppm
        self.isotope_ppm = isotope_ppm
        self.queue=queue
        self.reader_in, self.reader_out = reader_in, reader_out
        self.msn_rt_map = pd.Series(msn_rt_map)
        self.msn_rt_map.sort()
        self.results = results
        self.mass_labels = {'Light': {}} if mass_labels is None else mass_labels
        self.shifts = {0: "Light"}
        self.shifts.update({sum(silac_masses.keys()): silac_label for silac_label, silac_masses in six.iteritems(self.mass_labels)})
        self.raw_name = raw_name
        self.filename = os.path.split(self.raw_name)[1]
        self.rt_tol = 0.2 # for fitting
        self.debug = debug
        self.html = html
        self.mono = mono
        self.thread = thread
        self.fitting_run = fitting_run
        self.isotope_ppms = isotope_ppms
        self.quant_method = quant_method
        self.reporter_mode = reporter_mode
        self.spline = spline
        self.isotopologue_limit = isotopologue_limit
        self.labels_needed = labels_needed
        self.overlapping_mz = overlapping_mz
        self.min_resolution = min_resolution
        self.min_scans = min_scans
        self.quant_msn_map = quant_msn_map
        self.mrm = mrm
        self.mrm_pair_info = mrm_pair_info
        self.peak_cutoff = peak_cutoff
        self.replicate = replicate
        self.ratio_cutoff = 1
        self.ref_label = ref_label
        self.max_peaks = max_peaks
        self.parser_args = parser_args
        if mrm:
            self.quant_mrm_map = {label: list(group) for label, group in groupby(self.quant_msn_map, key=operator.itemgetter(0))}
        self.peaks_n = self.parser_args.peaks_n
        self.rt_guide = not self.parser_args.no_rt_guide
        self.filter_peaks = not self.parser_args.disable_peak_filtering
        self.report_ratios = not self.parser_args.no_ratios

    def get_calibrated_mass(self, mass):
        return mass/(1-self.spline(mass)/1e6) if self.spline else mass

    def flat_slope(self, combined_data, delta):
        return False
        data = combined_data.sum(axis='index').iloc[:10] if delta == -1 else combined_data.sum(axis='index').iloc[-10:]
        data /= combined_data.max().max()
        # remove the bottom values
        data = data[data>(data[data>data.quantile(0.5)].mean()*0.25)]
        slope, intercept, r_value, p_value, std_err = linregress(xrange(len(data)),data)
        return np.abs(slope) < 0.2 if self.mono else np.abs(slope) < 0.1

    # @line_profiler
    def replaceOutliers(self, common_peaks, combined_data, debug=False):
        x = []
        y = []
        tx = []
        ty = []
        ty2 = []
        hx = []
        hy = []
        keys = []
        hkeys = []
        y2 = []
        hy2 = []

        for i,v in common_peaks.items():
            for isotope, peaks in v.items():
                for peak_index, peak in enumerate(peaks):
                    keys.append((i, isotope, peak_index))
                    mean, std, std2 = peak['mean'], peak['std'], peak['std2']
                    x.append(mean)
                    y.append(std)
                    y2.append(std2)
                    if peak.get('valid'):
                        tx.append(mean)
                        ty.append(std)
                        ty2.append(std2)
                    if self.mrm and i != 'Light':
                        hx.append(mean)
                        hy.append(std)
                        hy2.append(std2)
                        hkeys.append((i, isotope, peak_index))
        classifier = EllipticEnvelope(support_fraction=0.75, random_state=0)
        if len(x) == 1:
            return x[0]
        data = np.array([x, y, y2]).T
        true_data = np.array([tx, ty, ty2]).T
        false_pred = (False, -1)
        true_pred = (True, 1)
        to_delete = set([])
        fitted = False
        true_data = np.vstack({tuple(row) for row in true_data}) if true_data.shape[0] else None
        if true_data is not None and true_data.shape[0] >= 3:
            fit_data = true_data
        else:
            fit_data = np.vstack({tuple(row) for row in data})

        if len(hx) >= 3 or fit_data.shape[0] >= 3:
            if debug:
                print(common_peaks)
            try:
                classifier.fit(np.array([hx, hy, hy2]).T if self.mrm else fit_data)
                fitted = True
                # x_mean, x_std1, x_std2 = classifier.location_
            except:
                try:
                    classifier = OneClassSVM(nu=0.95*0.15+0.05, kernel=str('linear'), degree=1, random_state=0)
                    classifier.fit(np.array([hx, hy, hy2]).T if self.mrm else fit_data)
                    fitted = True
                except:
                    if debug:
                        print(traceback.format_exc(), data)
        x_mean, x_std1, x_std2 = np.median(data, axis=0)
        if fitted:
            classes = classifier.predict(data)
            try:
                if hasattr(classifier, 'location_'):
                    x_mean, x_std1, x_std2 = classifier.location_
                else:
                    x_mean, x_std1, x_std2 = np.median(data[classes==1], axis=0)
            except IndexError:
                x_mean, x_std1, x_std2 = np.median(data, axis=0)
            else:
                x_inlier_indices = [i for i,v in enumerate(classes) if v in true_pred or common_peaks[keys[i][0]][keys[i][1]][keys[i][2]].get('valid')]
                x_inliers = set([keys[i][:2] for i in sorted(x_inlier_indices)])
                x_outliers = [i for i,v in enumerate(classes) if keys[i][:2] not in x_inliers and (v in false_pred or common_peaks[keys[i][0]][keys[i][1]][keys[i][2]].get('interpolate'))]
                if debug:
                    print('inliers', x_inliers)
                    print('outliers', x_outliers)
                # print('x1o', x1_outliers)
                min_x = x_mean-x_std1
                max_x = x_mean+x_std2
                for index in x_inlier_indices:
                    indexer = keys[index]
                    peak_info = common_peaks[indexer[0]][indexer[1]][indexer[2]]
                    peak_min = peak_info['mean']-peak_info['std']
                    peak_max = peak_info['mean']+peak_info['std2']
                    if peak_min < min_x:
                        min_x = peak_min
                    if peak_max > max_x:
                        max_x = peak_max
                if x_inliers:
                    for index in x_outliers:
                        indexer = keys[index]
                        if x_inliers is not None and indexer[:2] in x_inliers:
                            # this outlier has a valid inlying value in x1_inliers, so we delete it
                            to_delete.add(indexer)
                        else:
                            # there is no non-outlying data point. If this data point is > 1 sigma away, delete it
                            peak_info = common_peaks[indexer[0]][indexer[1]][indexer[2]]
                            if debug:
                                print(indexer, peak_info, x_mean, x_std1, x_std2)
                            if not (min_x < peak_info['mean'] < max_x):
                                to_delete.add(indexer)
        else:
            # we do not have enough data for ML, if we have scenarios with a 'valid' peak, keep them other others
            for quant_label,isotope_peaks in common_peaks.items():
                for isotope, peaks in isotope_peaks.items():
                    keys.append((i, isotope, peak_index))
                    to_keep = []
                    to_remove = []
                    for peak_index, peak in enumerate(peaks):
                        if peak.get('valid'):
                            to_keep.append(peak_index)
                        else:
                            to_remove.append(peak_index)
                    if to_keep:
                        for i in sorted(to_remove, reverse=True):
                            del peaks[i]

        if debug:
            print('to remove', to_delete)
        for i in sorted(set(to_delete), key=operator.itemgetter(0,1,2), reverse=True):
            del common_peaks[i[0]][i[1]][i[2]]
        return x_mean

    def convertScan(self, scan):
        import numpy as np
        scan_vals = scan['vals']
        res = pd.Series(scan_vals[:, 1].astype(np.uint64), index=np.round(scan_vals[:, 0], self.precision), name=int(scan['title']) if self.mrm else scan['rt'], dtype='uint64')
        # mz values can sometimes be not sorted -- rare but it happens
        res = res.sort_index()
        del scan_vals
        # due to precision, we have multiple m/z values at the same place. We can eliminate this by grouping them and summing them.
        # Summation is the correct choice here because we are combining values of a precision higher than we care about.
        try:
            return res.groupby(level=0).sum() if not res.empty else None
        except:
            print('Converting scan error {}\n{}\n{}\n'.format(traceback.format_exc(), res, scan))

    def getScan(self, ms1, start=None, end=None):
        self.reader_in.put((self.thread, ms1, start, end))
        scan = self.reader_out.get()
        if scan is None:
            print('Unable to fetch scan {}.\n'.format(ms1))
        return (self.convertScan(scan), {'centroid': scan.get('centroid', False)}) if scan is not None else (None, {})

    # @memory_profiler
    def quantify_peaks(self, params):
        try:
            html_images = {}
            scan_info = params.get('scan_info')
            target_scan = scan_info.get('id_scan')
            quant_scan = scan_info.get('quant_scan')
            scanId = target_scan.get('id')
            ms1 = quant_scan['id']
            scans_to_quant = quant_scan.get('scans')
            if scans_to_quant:
                scans_to_quant.pop(scans_to_quant.index(ms1))
            charge = target_scan['charge']
            mass = target_scan['mass']

            precursor = target_scan['precursor']
            calibrated_precursor = self.get_calibrated_mass(precursor)
            theor_mass = target_scan.get('theor_mass', calibrated_precursor)
            rt = target_scan['rt'] # this will be the RT of the target_scan, which is not always equal to the RT of the quant_scan

            peptide = target_scan.get('peptide')
            if self.debug:
                sys.stderr.write('thread {4} on ms {0} {1} {2} {3}\n'.format(ms1, rt, precursor, scan_info, id(self)))

            precursors = {}
            silac_dict = {'data': None, 'df': pd.DataFrame(), 'precursor': 'NA',
                          'isotopes': {}, 'peaks': OrderedDict(), 'intensity': 'NA'}
            data = OrderedDict()
            # data['Light'] = copy.deepcopy(silac_dict)
            combined_data = pd.DataFrame()
            highest_shift = 20
            if self.mrm:
                mrm_labels = [i for i in self.mrm_pair_info.columns if i.lower() not in ('retention time')]
                mrm_info = None
                for index, values in self.mrm_pair_info.iterrows():
                    if values['Light'] == mass:
                        mrm_info = values
            for silac_label, silac_masses in self.mass_labels.items():
                silac_shift=0
                global_mass = None
                added_residues = set([])
                cterm_mass = 0
                nterm_mass = 0
                mass_keys = list(silac_masses.keys())
                if self.reporter_mode:
                    silac_shift = sum(mass_keys)
                else:
                    if peptide:
                        for label_mass, label_masses in silac_masses.items():
                            if 'X' in label_masses:
                                global_mass = label_mass
                            if ']' in label_masses:
                                cterm_mass = label_mass
                            if '[' in label_masses:
                                nterm_mass = label_mass
                            added_residues = added_residues.union(label_masses)
                            labels = [label_mass for mod_aa in peptide if mod_aa in label_masses]
                            silac_shift += sum(labels)
                    else:
                        # no mass, just assume we have one of the labels
                        silac_shift += mass_keys[0]
                    if global_mass is not None:
                        silac_shift += sum([global_mass for mod_aa in peptide if mod_aa not in added_residues])
                    silac_shift += cterm_mass+nterm_mass
                    # get the non-specific ones
                    if silac_shift > highest_shift:
                        highest_shift = silac_shift
                precursors[silac_label] = silac_shift
                data[silac_label] = copy.deepcopy(silac_dict)
            if not precursors:
                precursors = {'Precursor': 0.0}
            precursors = OrderedDict(sorted(precursors.items(), key=operator.itemgetter(1)))
            shift_maxes = {i: j for i,j in zip(precursors.keys(), list(precursors.values())[1:])}
            finished_isotopes = {i: set([]) for i in precursors.keys()}
            result_dict = {'peptide': target_scan.get('mod_peptide', peptide),
                           'scan': scanId, 'ms1': ms1, 'charge': charge,
                           'modifications': target_scan.get('modifications'), 'rt': rt,
                           'accession': target_scan.get('accession')}
            ms_index = 0
            delta = -1
            theo_dist = peaks.calculate_theoretical_distribution(peptide.upper()) if peptide else None
            spacing = config.NEUTRON/float(charge)
            isotope_labels = {}
            isotopes_chosen = {}
            last_precursors = {-1: {}, 1: {}}
            # our rt might sometimes be an approximation, such as from X!Tandem which requires some transformations
            initial_scan = find_scan(self.quant_msn_map, ms1)
            current_scan = None
            scans_to_skip = set([])
            not_found = 0
            if self.mrm:
                mrm_label = mrm_labels.pop() if mrm_info is not None else 'Light'
                mass = mass if mrm_info is None else mrm_info[mrm_label]
            last_peak_height = {i: defaultdict(int) for i in precursors.keys()}
            low_int_isotopes = defaultdict(int)
            while True:
                map_to_search = self.quant_mrm_map[mass] if self.mrm else self.quant_msn_map
                if current_scan is None:
                    current_scan = initial_scan
                else:
                    if scans_to_quant:
                        current_scan = scans_to_quant.pop()
                    elif scans_to_quant is None:
                        current_scan = find_prior_scan(map_to_search, current_scan) if delta == -1 else find_next_scan(map_to_search, current_scan)
                    else:
                        # we've exhausted the scans we are supposed to quantify
                        break
                found = set([])
                if current_scan is not None:
                    if current_scan in scans_to_skip:
                        continue
                    else:
                        df, scan_params = self.getScan(current_scan, start=None if self.mrm else precursor-5, end=None if self.mrm else precursor+highest_shift)
                        # check if it's a low res scan, if so skip it
                        if self.min_resolution and df is not None:
                            scan_resolution = np.average(df.index[1:]/np.array([df.index[i]-df.index[i-1] for i in xrange(1,len(df))]))
                            # print self.msn_rt_map.index[next_scan], self.min_resolution, scan_resolution
                            if scan_resolution < self.min_resolution:
                                scans_to_skip.add(current_scan)
                                continue
                    if df is not None:
                        labels_found = set([])
                        xdata = df.index.values.astype(float)
                        ydata = df.fillna(0).values.astype(float)
                        iterator = precursors.items() if not self.mrm else [(mrm_label, 0)]
                        for precursor_label, precursor_shift in iterator:
                            selected = {}
                            if self.mrm:
                                labels_found.add(precursor_label)
                                for i,j in zip(xdata, ydata):
                                    selected[i] = j
                                isotope_labels[df.name] = {'label': precursor_label, 'isotope_index': target_scan.get('product_ion', 0)}
                                key = (df.name, i)
                                isotopes_chosen[key] = {'label': precursor_label, 'isotope_index': target_scan.get('product_ion', 0), 'amplitude': j}
                            else:
                                if self.reporter_mode:
                                    measured_precursor = precursor_shift
                                    uncalibrated_precursor = precursor_shift
                                    theoretical_precursor = precursor_shift
                                else:
                                    uncalibrated_precursor = precursor+precursor_shift/float(charge)
                                    measured_precursor = self.get_calibrated_mass(uncalibrated_precursor)
                                    theoretical_precursor = theor_mass+precursor_shift/float(charge)
                                data[precursor_label]['calibrated_precursor'] = measured_precursor
                                data[precursor_label]['precursor'] = uncalibrated_precursor
                                shift_max = shift_maxes.get(precursor_label)
                                shift_max = self.get_calibrated_mass(precursor+shift_max/float(charge)) if shift_max is not None and self.overlapping_mz is False else None
                                is_fragmented_scan = (current_scan == initial_scan) and (precursor == measured_precursor)
                                envelope = peaks.findEnvelope(xdata, ydata, measured_mz=measured_precursor, theo_mz=theoretical_precursor, max_mz=shift_max,
                                                              charge=charge, precursor_ppm=self.precursor_ppm, isotope_ppm=self.isotope_ppm, reporter_mode=self.reporter_mode,
                                                              isotope_ppms=self.isotope_ppms if self.fitting_run else None, quant_method=self.quant_method, debug=self.debug,
                                                              theo_dist=theo_dist if self.mono or precursor_shift == 0.0 else None, label=precursor_label, skip_isotopes=finished_isotopes[precursor_label],
                                                              last_precursor=last_precursors[delta].get(precursor_label, measured_precursor),
                                                              isotopologue_limit=self.isotopologue_limit, fragment_scan=is_fragmented_scan,
                                                              centroid=scan_params.get('centroid', False))
                                if not envelope['envelope']:
                                    if self.debug:
                                        print('envelope empty', envelope, measured_precursor, initial_scan, current_scan, last_precursors)
                                    if self.parser_args.msn_all_scans:
                                        selected[measured_precursor] = 0
                                        isotope_labels[measured_precursor] = {'label': precursor_label, 'isotope_index': 0}
                                        isotopes_chosen[(df.name, measured_precursor)] = {'label': precursor_label, 'isotope_index': 0, 'amplitude': 0}
                                    else:
                                        continue

                                if not self.parser_args.msn_all_scans and 0 in envelope['micro_envelopes'] and envelope['micro_envelopes'][0].get('int'):
                                    if ms_index == 0:
                                        last_precursors[delta*-1][precursor_label] = envelope['micro_envelopes'][0]['params'][1]
                                    last_precursors[delta][precursor_label] = envelope['micro_envelopes'][0]['params'][1]
                                added_keys = []
                                for isotope, vals in six.iteritems(envelope['micro_envelopes']):
                                    if isotope in finished_isotopes[precursor_label]:
                                        continue
                                    peak_intensity = vals.get('int')
                                    if peak_intensity == 0 or (self.peak_cutoff and peak_intensity < last_peak_height[precursor_label][isotope]*self.peak_cutoff):
                                        low_int_isotopes[(precursor_label, isotope)] += 1
                                        if low_int_isotopes[(precursor_label, isotope)] >= 2:
                                            if self.debug:
                                                print('finished with isotope', precursor_label, envelope)
                                            finished_isotopes[precursor_label].add(isotope)
                                        else:
                                            labels_found.add(precursor_label)
                                        continue
                                    else:
                                        low_int_isotopes[(precursor_label, isotope)] = 0
                                        found.add(precursor_label)
                                        labels_found.add(precursor_label)
                                    if current_scan == initial_scan or last_peak_height[precursor_label][isotope] == 0:
                                        last_peak_height[precursor_label][isotope] = peak_intensity
                                    selected[measured_precursor+isotope*spacing] = peak_intensity
                                    vals['isotope'] = isotope
                                    isotope_labels[measured_precursor+isotope*spacing] = {'label': precursor_label, 'isotope_index': isotope}
                                    key = (df.name, measured_precursor+isotope*spacing)
                                    added_keys.append(key)
                                    isotopes_chosen[key] = {'label': precursor_label, 'isotope_index': isotope, 'amplitude': peak_intensity}
                                del envelope
                            selected = pd.Series(selected, name=df.name).to_frame()
                            if df.name in combined_data.columns:
                                combined_data = combined_data.add(selected, axis='index', fill_value=0)
                            else:
                                combined_data = pd.concat([combined_data, selected], axis=1).fillna(0)
                            del selected
                        if not self.mrm and (not self.parser_args.msn_all_scans and len(labels_found) < self.labels_needed):
                            found.discard(precursor_label)
                            if df is not None and df.name in combined_data.columns:
                                del combined_data[df.name]
                                for i in isotopes_chosen.keys():
                                    if i[0] == df.name:
                                        del isotopes_chosen[i]
                        del df

                if not found or (np.abs(ms_index) > 7 and self.flat_slope(combined_data, delta)):
                    not_found += 1
                    if current_scan is None or (not_found >= 2 and not self.parser_args.msn_all_scans):
                        not_found = 0
                        if delta == -1:
                            delta = 1
                            current_scan = initial_scan
                            finished_isotopes = {i: set([]) for i in precursors.keys()}
                            last_peak_height = {i: defaultdict(int) for i in precursors.keys()}
                            ms_index = 0
                        else:
                            if self.mrm:
                                if mrm_info is not None and mrm_labels:
                                    mrm_label = mrm_labels.pop() if mrm_info is not None else 'Light'
                                    mass = mass if mrm_info is None else mrm_info[mrm_label]
                                    delta = -1
                                    current_scan = self.quant_mrm_map[mass][0][1]
                                    last_peak_height = {i: defaultdict(int) for i in precursors.keys()}
                                    initial_scan = current_scan
                                    finished_isotopes = {i: set([]) for i in precursors.keys()}
                                    ms_index = 0
                                else:
                                    break
                            else:
                                break
                else:
                    not_found = 0
                if self.reporter_mode:
                    break
                ms_index += delta
            rt_figure = {}
            isotope_figure = {}
            if self.parser_args.merge_labels:
                combined_data = combined_data.sum(axis=0).to_frame(name=combined_data.index[0]).T
            if isotopes_chosen and isotope_labels and not combined_data.empty:
                if self.mrm:
                    combined_data = combined_data.T
                # bookend with zeros if there aren't any, do the right end first because pandas will by default append there
                combined_data = combined_data.sort_index().sort_index(axis='columns')
                start_rt = rt
                if len(combined_data.columns) == 1:
                    try:
                        new_col = self.msn_rt_map.iloc[self.msn_rt_map.searchsorted(combined_data.columns[-1])+1].values[0]
                    except:
                        if self.debug:
                            print(combined_data.columns)
                            print(self.msn_rt_map)
                else:
                    new_col = combined_data.columns[-1]+(combined_data.columns[-1]-combined_data.columns[-2])
                combined_data[new_col] = 0
                new_col = combined_data.columns[0]-(combined_data.columns[1]-combined_data.columns[0])
                combined_data[new_col] = 0
                combined_data = combined_data[sorted(combined_data.columns)]

                combined_data = combined_data.sort_index().sort_index(axis='columns')
                quant_vals = defaultdict(dict)
                isotope_labels = pd.DataFrame(isotope_labels).T

                isotopes_chosen = pd.DataFrame(isotopes_chosen).T
                isotopes_chosen.index.names = ['RT', 'MZ']

                if self.html:
                    # make the figure of our isotopes selected
                    all_x = sorted(isotopes_chosen.index.get_level_values('MZ').drop_duplicates())
                    isotopes_chosen['RT'] = isotopes_chosen.index.get_level_values('RT')
                    isotope_group = isotopes_chosen.groupby('RT')

                    isotope_figure = {
                        'data': [],
                        'plot-multi': True,
                        'common-x': ['x']+all_x,
                        'max-y': isotopes_chosen['amplitude'].max(),
                    }
                    isotope_figure_mapper = {}
                    rt_figure = {
                        'data': [],
                        'plot-multi': True,
                        'common-x': ['x']+['{0:0.4f}'.format(i) for i in combined_data.columns],
                        'rows': len(precursors),
                        'max-y': combined_data.max().max(),
                    }
                    rt_figure_mapper = {}

                    for counter, (index, row) in enumerate(isotope_group):
                        try:
                            title = 'Scan {} RT {}'.format(self.msn_rt_map[self.msn_rt_map==index].index[0], index)
                        except:
                            title = '{}'.format(index)
                        if index in isotope_figure_mapper:
                            isotope_base = isotope_figure_mapper[index]
                        else:
                            isotope_base = {'data': {'x': 'x', 'columns': [], 'type': 'bar'}, 'axis': {'x': {'label': 'M/Z'}, 'y': {'label': 'Intensity'}}}
                            isotope_figure_mapper[index] = isotope_base
                            isotope_figure['data'].append(isotope_base)
                        for group in precursors.keys():
                            label_df = row[row['label'] == group]
                            x = label_df['amplitude'].index.get_level_values('MZ').tolist()
                            y = label_df['amplitude'].values.tolist()
                            isotope_base['data']['columns'].append(['{} {}'.format(title, group)]+[y[x.index(i)] if i in x else 0 for i in all_x])
                if not self.reporter_mode:
                    combined_peaks = defaultdict(dict)

                    merged_data = combined_data.sum(axis=0)
                    rt_attempts = 0
                    found_rt = False
                    merged_x = merged_data.index.astype(float).values
                    merged_y = merged_data.values.astype(float)
                    fitting_y = np.copy(merged_y)
                    mval = merged_y.max()
                    while rt_attempts < 4 and not found_rt:
                        if self.debug:
                            print('MERGED PEAK FINDING')
                        res, residual = peaks.findAllPeaks(merged_x, fitting_y, filter=False, bigauss_fit=True,
                                                           rt_peak=start_rt, max_peaks=self.max_peaks, debug=self.debug,
                                                           snr=self.parser_args.snr_filter, amplitude_filter=self.parser_args.intensity_filter)
                        rt_peak = peaks.bigauss_ndim(np.array([rt]), res)[0]
                        # we don't do this routine for cases where there are > 5
                        found_rt = (not self.rt_guide and self.parser_args.msn_all_scans) or sum(fitting_y>0) <= 5 or rt_peak > 0.05
                        if not found_rt and rt_peak < 0.05:
                            # get the closest peak
                            nearest_peak = sorted([(i, np.abs(rt-i)) for i in res[1::4]], key=operator.itemgetter(1))[0][0]
                            # this is tailored to massa spectrometry elution profiles at the moment, and only evaluates for situtations where the rt and peak
                            # are no further than a minute apart.
                            if np.abs(nearest_peak-rt) < 1:
                                rt_index, peak_index = peaks.find_nearest_indices(merged_x, [rt, nearest_peak])
                                if rt_index < 0:
                                    rt_index = 0
                                if peak_index == -1:
                                    peak_index = len(fitting_y)
                                if rt_index != peak_index:
                                    grad_len = np.abs(peak_index-rt_index)
                                    if grad_len < 4:
                                        found_rt = True
                                    else:
                                        gradient = (np.gradient(fitting_y[rt_index:peak_index])>0) if rt_index < peak_index else (np.gradient(fitting_y[peak_index:rt_index])<0)
                                        if sum(gradient) >= grad_len-1:
                                            found_rt = True
                                else:
                                    found_rt = True
                        if not found_rt:
                            if self.debug:
                                print('cannot find rt for', peptide, rt_peak)
                                print(merged_x, fitting_y, res, sum(fitting_y>0))
                            # destroy our peaks, keep searching
                            res[::4] = res[::4]*fitting_y.max()
                            fitting_y -= peaks.bigauss_ndim(merged_x, res)
                            fitting_y[fitting_y<0] = 0
                        rt_attempts += 1
                    if not found_rt and self.debug:
                        print(peptide, 'is dead', rt_attempts, found_rt)
                    elif self.debug:
                        print('peak used for sub-fitting', res)
                    if found_rt:
                        rt_means = res[1::4]
                        rt_amps = res[::4]
                        rt_std = res[2::4]
                        rt_std2 = res[3::4]
                        m_std = np.std(merged_y)
                        m_mean = np.mean(merged_y)
                        valid_peaks = [{'mean': i, 'amp': j*mval, 'std': l, 'std2': k, 'total': merged_y.sum(), 'snr': m_mean/m_std, 'residual': residual}
                                                for i, j, l, k in zip(rt_means, rt_amps, rt_std, rt_std2)]
                        # if we have a peaks containing our retention time, keep them and throw out ones not containing it
                        # to_remove = []
                        # to_keep = []
                        # print valid_peaks
                        # for i,v in enumerate(valid_peaks):
                        #     mu = v['mean']
                        #     s1 = v['std']
                        #     s2 = v['std2']
                        #     if mu-s1*3 < start_rt < mu+s2*3:
                        #         v['valid'] = True
                        #         to_keep.append(i)
                        #     else:
                        #         to_remove.append(i)
                        # # kick out peaks not containing our RT
                        # valid_peak = None
                        # if not to_keep:
                        #     # we have no peaks with our RT, there are contaminating peaks, remove all the noise but the closest to our RT
                        #     if not self.mrm:
                        #         valid_peak = sorted([(i, np.abs(i['mean']-start_rt)) for i in valid_peaks], key=operator.itemgetter(1))[0][0]
                        #         valid_peak['interpolate'] = True
                        #     # else:
                        #     #     valid_peak = [j[0] for j in sorted([(i, i['amp']) for i in valid_peaks], key=operator.itemgetter(1), reverse=True)[:3]]
                        # else:
                        #     for i in reversed(to_remove):
                        #         del valid_peaks[i]
                        #     # sort by RT
                        if self.rt_guide:
                            valid_peaks.sort(key=lambda x: np.abs(x['mean']-start_rt))

                            peak_index = peaks.find_nearest_index(merged_x, valid_peaks[0]['mean'])
                            peak_location = merged_x[peak_index]
                            if self.debug:
                                print('peak location is', peak_location)
                            merged_lb = peaks.find_nearest_index(merged_x, valid_peaks[0]['mean']-valid_peaks[0]['std']*2)
                            merged_rb = peaks.find_nearest_index(merged_x, valid_peaks[0]['mean']+valid_peaks[0]['std2']*2)
                            merged_rb = len(merged_x) if merged_rb == -1 else merged_rb+1
                        else:
                            merged_lb = 0
                            merged_rb = len(merged_x)

                        for row_num, (index, values) in enumerate(combined_data.iterrows()):
                            quant_label = isotope_labels.loc[index, 'label']
                            xdata = values.index.values.astype(float)
                            ydata = values.fillna(0).values.astype(float)
                            if sum(ydata>0) >= self.min_scans:
                                # this step is to add in a term on the border if possible
                                # otherwise, there are no penalties on the variance if it is
                                # at the border since the data does not exist. We only add for lower values to avoid
                                # including monster peaks we may be explicitly excluding above
                                fit_lb = merged_lb
                                fit_rb = merged_rb
                                while fit_rb+1 < len(ydata) and ydata[fit_rb+1] <= ydata[fit_rb-1]:
                                    fit_rb += 1
                                while fit_lb != 0 and ydata[fit_lb] >= ydata[fit_lb-1]:
                                    fit_lb -= 1
                                peak_x = np.copy(xdata[fit_lb:fit_rb])
                                peak_y = np.copy(ydata[fit_lb:fit_rb])
                                if peak_x.size <= 1 or sum(peak_y>0) < self.min_scans:
                                    continue
                                if self.rt_guide:
                                    peak_positive_y = peak_y>0
                                    nearest_positive_peak = peaks.find_nearest(peak_x[peak_positive_y], peak_location)
                                    sub_peak_location = peaks.find_nearest_index(peak_x, nearest_positive_peak)
                                    sub_peak_index = sub_peak_location if peak_y[sub_peak_location] else np.argmax(peak_y)
                                else:
                                    nearest_positive_peak = 0
                                # fit, residual = peaks.fixedMeanFit2(peak_x, peak_y, peak_index=sub_peak_index, debug=self.debug)
                                if self.debug:
                                    print('fitting XIC for', quant_label, index)
                                fit, residual = peaks.findAllPeaks(xdata, ydata, bigauss_fit=True, filter=self.filter_peaks, max_peaks=self.max_peaks,
                                                                   rt_peak=nearest_positive_peak, debug=self.debug, peak_width_start=1,
                                                                   snr=self.parser_args.snr_filter, amplitude_filter=self.parser_args.intensity_filter,
                                                                   peak_width_end=self.parser_args.min_peak_separation)
                                if fit is None:
                                    continue
                                rt_means = fit[1::4]
                                rt_amps = fit[::4]*ydata.max()
                                rt_std = fit[2::4]
                                rt_std2 = fit[3::4]
                                xic_peaks = []
                                positive_y = ydata[ydata>0]
                                if len(positive_y) > 5:
                                    positive_y = gaussian_filter1d(positive_y, 3, mode='constant')
                                for i, j, l, k in zip(rt_means, rt_amps, rt_std, rt_std2):
                                    d = {'mean': i, 'amp': j, 'std': l, 'std2': k, 'total': values.sum(), 'residual': residual}
                                    mean_index = peaks.find_nearest_index(xdata[ydata>0], i)
                                    window_size = 5 if len(positive_y) < 15 else len(positive_y)/3
                                    lb, rb = mean_index-window_size, mean_index+window_size+1
                                    if lb < 0:
                                        lb = 0
                                    if rb > len(positive_y):
                                        rb = -1
                                    data_window = positive_y[lb:rb]
                                    try:
                                        background = np.percentile(data_window, 0.8)
                                    except:
                                        background = np.percentile(ydata, 0.8)
                                    mean = np.mean(data_window)
                                    if background < mean:
                                        background = mean
                                    d['sbr'] = np.mean(j/(np.array(sorted(data_window, reverse=True)[:5])))#(j-np.mean(positive_y[lb:rb]))/np.std(positive_y[lb:rb])
                                    d['snr'] = (j-background)/np.std(data_window)
                                    xic_peaks.append(d)
                                # if we have a peaks containing our retention time, keep them and throw out ones not containing it
                                to_remove = []
                                to_keep = []
                                if self.rt_guide:
                                    for i,v in enumerate(xic_peaks):
                                        mu = v['mean']
                                        s1 = v['std']
                                        s2 = v['std2']
                                        if mu-s1*2 < start_rt < mu+s2*2:
                                            # these peaks are considered true and will help with the machine learning
                                            if mu-s1*1.5 < start_rt < mu+s2*1.5:
                                                v['valid'] = True
                                                to_keep.append(i)
                                        elif peaks.find_nearest_index(merged_x, mu)-peak_location > 2:
                                            to_remove.append(i)
                                # kick out peaks not containing our RT
                                if self.rt_guide:
                                    if not to_keep:
                                        # we have no peaks with our RT, there are contaminating peaks, remove all the noise but the closest to our RT
                                        if not self.mrm:
                                            # for i in to_remove:
                                            #     xic_peaks[i]['interpolate'] = True
                                            valid_peak = sorted([(i, np.abs(i['mean']-start_rt)) for i in xic_peaks], key=operator.itemgetter(1))[0][0]
                                            for i in reversed(xrange(len(xic_peaks))):
                                                if xic_peaks[i] == valid_peak:
                                                    continue
                                                else:
                                                    del xic_peaks[i]
                                            # valid_peak['interpolate'] = True
                                        # else:
                                        #     valid_peak = [j[0] for j in sorted([(i, i['amp']) for i in xic_peaks], key=operator.itemgetter(1), reverse=True)[:3]]
                                    else:
                                        # if not to_remove:
                                        #     xic_peaks = [xic_peaks[i] for i in to_keep]
                                        # else:
                                        for i in reversed(to_remove):
                                            del xic_peaks[i]
                                if self.debug:
                                    print(quant_label, index)
                                    print(fit)
                                    print(to_remove,to_keep, xic_peaks)
                                combined_peaks[quant_label][index] = xic_peaks# if valid_peak is None else [valid_peak]

                            if self.html:
                                # ax = fig.add_subplot(subplot_rows, subplot_columns, fig_index)
                                if quant_label in rt_figure_mapper:
                                    rt_base = rt_figure_mapper[(quant_label, index)]
                                else:
                                    rt_base = {'data': {'x': 'x', 'columns': []}, 'grid': {'x': {'lines': [{'value': rt, 'text': 'Initial RT {0:0.4f}'.format(rt), 'position': 'middle'}]}}, 'subchart': {'show': True}, 'axis': {'x': {'label': 'Retention Time'}, 'y': {'label': 'Intensity'}}}
                                    rt_figure_mapper[(quant_label, index)] = rt_base
                                    rt_figure['data'].append(rt_base)
                                rt_base['data']['columns'].append(['{0} {1} raw'.format(quant_label, index)]+ydata.tolist())

                peak_info = {i: {} for i in self.mrm_pair_info.columns} if self.mrm else {i: {} for i in data.keys()}
                if self.reporter_mode or combined_peaks:
                    if self.reporter_mode:
                        for row_num, (index, values) in enumerate(combined_data.iterrows()):
                            quant_label = isotope_labels.loc[index, 'label']
                            isotope_index = isotope_labels.loc[index, 'isotope_index']
                            int_val = sum(values)
                            quant_vals[quant_label][isotope_index] = int_val
                    else:
                        common_peak = self.replaceOutliers(combined_peaks, combined_data, debug=self.debug)
                        common_loc = peaks.find_nearest_index(xdata, common_peak)#np.where(xdata==common_peak)[0][0]
                        for quant_label, quan_values in combined_peaks.items():
                            for index, values in quan_values.items():
                                if not values:
                                    continue
                                isotope_index = isotope_labels.loc[index, 'isotope_index']
                                rt_values = combined_data.loc[index]
                                xdata = rt_values.index.values.astype(float)
                                ydata = rt_values.fillna(0).values.astype(float)
                                # pick the biggest within a rt cutoff of 0.2, otherwise pick closest
                                # closest_rts = sorted([(i, i['amp']) for i in values if np.abs(i['peak']-common_peak) < 0.2], key=operator.itemgetter(1), reverse=True)
                                # if not closest_rts:
                                # print values
                                closest_rts = sorted([(i, np.abs(i['mean']-common_peak)) for i in values], key=operator.itemgetter(1))
                                xic_peaks = [i[0] for i in closest_rts]
                                pos_x = xdata[ydata>0]
                                if self.rt_guide:
                                    xic_peaks = [xic_peaks[0]]
                                else:
                                    # unguided, sort by amplitude
                                    xic_peaks.sort(key=operator.itemgetter('amp'), reverse=True)
                                for xic_peak_index, xic_peak in enumerate(xic_peaks):
                                    if self.peaks_n != -1 and xic_peak_index > self.peaks_n:
                                        break
                                    # if we move more than a # of ms1 to the dominant peak, update to our known peak
                                    gc = 'k'
                                    nearest = peaks.find_nearest_index(pos_x, xic_peak['mean'])
                                    peak_loc = np.where(xdata==pos_x[nearest])[0][0]
                                    mean = xic_peak['mean']
                                    amp = xic_peak['amp']
                                    mean_diff = mean-xdata[common_loc]
                                    mean_diff = np.abs(mean_diff/xic_peak['std'] if mean_diff < 0 else mean_diff/xic_peak['std2'])
                                    std = xic_peak['std']
                                    std2 = xic_peak['std2']
                                    snr = xic_peak['snr']
                                    sbr = xic_peak['sbr']
                                    residual = xic_peak['residual']
                                    if False and len(xdata) >= 3 and (mean_diff > 2 or (np.abs(peak_loc-common_loc) > 2 and mean_diff > 2)):
                                        # fixed mean fit
                                        if self.debug:
                                            print(quant_label, index)
                                            print(common_loc, peak_loc)
                                        nearest = peaks.find_nearest_index(pos_x, mean)
                                        nearest_index = np.where(xdata==pos_x[nearest])[0][0]
                                        res = peaks.fixedMeanFit(xdata, ydata, peak_index=nearest_index, debug=self.debug)
                                        if res is None:
                                            if self.debug:
                                                print(quant_label, index, 'has no values here')
                                            continue
                                        amp, mean, std, std2 = res
                                        amp *= ydata.max()
                                        gc = 'g'
                                    #var_rat = closest_rt['var']/common_var
                                    peak_params = np.array([amp,  mean, std, std2])
                                    # int_args = (res.x[rt_index]*mval, res.x[rt_index+1], res.x[rt_index+2])
                                    left, right = xdata[0]-4*std, xdata[-1]+4*std2
                                    xr = np.linspace(left, right, 1000)
                                    left_index, right_index = peaks.find_nearest_index(xdata, left), peaks.find_nearest_index(xdata, right)+1
                                    if left_index < 0:
                                        left_index = 0
                                    if right_index >= len(xdata) or right_index <= 0:
                                        right_index = len(xdata)
                                    try:
                                        int_val = integrate.simps(peaks.bigauss_ndim(xr, peak_params), x=xr) if self.quant_method == 'integrate' else ydata[(xdata > left) & (xdata < right)].sum()
                                    except:
                                        if self.debug:
                                            print(traceback.format_exc())
                                            print(xr, peak_params)
                                    try:
                                        total_int = integrate.simps(ydata[left_index:right_index], x=xdata[left_index:right_index])
                                    except:
                                        if self.debug:
                                            print(traceback.format_exc())
                                            print(left_index, right_index, xdata, ydata)
                                    sdr = np.log2(int_val*1./total_int+1.)

                                    if int_val and not pd.isnull(int_val) and gc != 'c':
                                        try:
                                            quant_vals[quant_label][isotope_index] += int_val
                                        except KeyError:
                                            try:
                                                quant_vals[quant_label][isotope_index] = int_val
                                            except KeyError:
                                                quant_vals[quant_label] = {isotope_index: int_val}
                                    peak_info_dict = {'mean': mean, 'std': std, 'std2': std2, 'amp': amp,
                                                      'mean_diff': mean_diff, 'snr': snr, 'sbr': sbr, 'sdr': sdr,
                                                      'auc': int_val, 'peak_width': std+std2}
                                    try:
                                        peak_info[quant_label][isotope_index][xic_peak_index] = peak_info_dict
                                    except KeyError:
                                        try:
                                            peak_info[quant_label][isotope_index] = {xic_peak_index: peak_info_dict}
                                        except KeyError:
                                            peak_info[quant_label] = {isotope_index: {xic_peak_index: peak_info_dict}}

                                    try:
                                        data[quant_label]['residual'].append(residual)
                                    except KeyError:
                                        data[quant_label]['residual'] = [residual]

                                    if self.html:
                                        rt_base = rt_figure_mapper[(quant_label, index)]
                                        key = '{} {}'.format(quant_label, index)
                                        for i,v in enumerate(rt_base['data']['columns']):
                                            if key in v[0]:
                                                break
                                        rt_base['data']['columns'].insert(i, ['{0} {1} fit {2}'.format(quant_label, index, xic_peak_index)]+np.nan_to_num(peaks.bigauss_ndim(xdata, peak_params)).tolist())
                        del combined_peaks
                write_html = True if self.ratio_cutoff == 0 else False

                # # Some experimental code that tries to compare the XIC with the theoretical distribution
                # # Currently disabled as it reduces the number of datapoints to infer SILAC ratios and results in poorer
                # # comparisons -- though there might be merit to intensity based estimates with this.
                if self.parser_args.theo_xic and self.mono and theo_dist is not None:
                    # Compare the extracted XIC with the theoretical abundance of each isotope:
                    # To do this, we take the residual of all combinations of isotopes
                    for quant_label in quant_vals:
                        isotopes = quant_vals[quant_label].keys()
                        isotope_ints = {i: quant_vals[quant_label][i] for i in isotopes}
                        isotope_residuals = []
                        for num_of_isotopes in xrange(2, len(isotopes)+1):
                            for combo in combinations(isotopes, num_of_isotopes):
                                chosen_isotopes = np.array([isotope_ints[i] for i in combo])
                                chosen_isotopes /= chosen_isotopes.max()
                                chosen_dist = np.array([theo_dist[i] for i in combo])
                                chosen_dist /= chosen_dist.max()
                                res = sum((chosen_dist-chosen_isotopes)**2)
                                isotope_residuals.append((res, combo))
                        # this weird sorting is to put the favorable values as the lowest values
                        if isotope_residuals:
                            kept_keys = sorted(isotope_residuals, key=lambda x: (0 if x[0]<0.1 else 1, len(isotopes)-len(x[1]), x[0]))[0][1]
                            # print(quant_label, kept_keys)
                            for i in isotopes:
                                if i not in kept_keys:
                                    del quant_vals[quant_label][i]

                for silac_label1 in data.keys():
                    # TODO: check if peaks overlap before taking ratio
                    qv1 = quant_vals.get(silac_label1, {})
                    result_dict.update({
                        '{}_intensity'.format(silac_label1): sum(qv1.values())
                    })
                    if self.report_ratios:
                        for silac_label2 in data.keys():
                            if self.ref_label is not None and str(silac_label2.lower()) != self.ref_label.lower():
                                continue
                            if silac_label1 == silac_label2:
                                continue
                            qv2 = quant_vals.get(silac_label2, {})
                            ratio = 'NA'
                            if qv1 is not None and qv2 is not None:
                                if self.mono:
                                    common_isotopes = set(qv1.keys()).intersection(qv2.keys())
                                    x = []
                                    y = []
                                    l1, l2 = 0,0
                                    for i in common_isotopes:
                                        q1 = qv1.get(i)
                                        q2 = qv2.get(i)
                                        if q1 > 100 and q2 > 100 and q1 > l1*0.15 and q2 > l2*0.15:
                                            x.append(i)
                                            y.append(q1/q2)
                                            l1, l2 = q1, q2
                                    # fit it and take the intercept
                                    if len(x) >= 3 and np.std(np.log2(y))>0.3:
                                        classifier = EllipticEnvelope(contamination=0.25, random_state=0)
                                        fit_data = np.log2(np.array(y).reshape(len(y),1))
                                        true_pred = (True, 1)
                                        classifier.fit(fit_data)
                                        #print peptide, ms1, np.mean([y[i] for i,v in enumerate(classifier.predict(fit_data)) if v in true_pred]), y
                                        ratio = np.mean([y[i] for i,v in enumerate(classifier.predict(fit_data)) if v in true_pred])
                                    else:
                                        ratio = np.array(y).mean()
                                else:
                                    common_isotopes = set(qv1.keys()).union(qv2.keys())
                                    quant1 = sum([qv1.get(i, 0) for i in common_isotopes])
                                    quant2 = sum([qv2.get(i, 0) for i in common_isotopes])
                                    ratio = quant1/quant2 if quant1 and quant2 else 'NA'
                                try:
                                    if np.abs(np.log2(ratio)) > self.ratio_cutoff:
                                        write_html = True
                                except:
                                    pass
                            result_dict.update({'{}_{}_ratio'.format(silac_label1, silac_label2): ratio})

                if write_html:
                    result_dict.update({'html_info': html_images})
                for peak_label, peak_data in six.iteritems(peak_info):
                    # all_xic_peak_info = []
                    # for peak_isotope, isotope_peaks in six.iteritems(peak_data):
                    #     for xic_index, xic_peak_info in six.iteritems(peak_data):
                    #         all_xic_peak_info.append(xic_peak_info)
                            # w1, w2 = xic_peak_info.get('std', None), xic_peak_info.get('std2', None)
                            # all_xic_peak_info.append({
                            #     'peak_intensity': xic_peak_info.get('amp', 'NA'),
                            #     'auc': xic_peak_info.get('auc', 'NA'),
                            #     'mean': xic_peak_info.get('mean', 'NA'),
                            #     'snr': np.mean(pd.Series(xic_peak_info.get('snr', [])).replace([np.inf, -np.inf, np.nan], 0)),
                            #     'sbr': np.mean(pd.Series(xic_peak_info.get('sbr', [])).replace([np.inf, -np.inf, np.nan], 0)),
                            #     'sdr': np.mean(pd.Series(xic_peak_info.get('sdr', [])).replace([np.inf, -np.inf, np.nan], 0)),
                            #     'rt_width': w1+w2 if w1 and w2 else 'NA',
                            #     'mean_diff': np.mean(pd.Series(xic_peak_info.get('mean_diff', [])).replace([np.inf, -np.inf, np.nan], 0))
                            # })
                    result_dict.update({
                        '{}_peaks'.format(peak_label): peak_data,
                        '{}_isotopes'.format(peak_label): sum((isotopes_chosen['label'] == peak_label) & (isotopes_chosen['amplitude']>0)),
                    })
            if self.parser_args.merge_labels:
                merged_precursor = ','.join((str(v['precursor']) for v in data.values()))
                merged_calprecursor = ','.join((str(v.get('calibrated_precursor', v['precursor'])) for v in data.values()))
            for silac_label, silac_data in six.iteritems(data):
                precursor = merged_precursor if self.parser_args.merge_labels else silac_data['precursor']
                calc_precursor = merged_calprecursor if self.parser_args.merge_labels else silac_data.get('calibrated_precursor', silac_data['precursor'])
                result_dict.update({
                    '{}_residual'.format(silac_label): np.mean(pd.Series(silac_data.get('residual', [])).replace([np.inf, -np.inf, np.nan], 0)),
                    '{}_precursor'.format(silac_label): precursor,
                    '{}_calibrated_precursor'.format(silac_label): calc_precursor,
                })
            result_dict.update({
                'ions_found': target_scan.get('ions_found'),
                'html': {'xic': rt_figure, 'isotope': isotope_figure}
            })
            self.results.put(result_dict)
            del result_dict
            del data
            del combined_data
            del isotopes_chosen
        except:
            # if self.debug:
            print('ERROR ON {}: {}'.format(peptide, traceback.format_exc()))
            try:
                self.results.put(result_dict)
            except:
                pass
            return

    def run(self):
        for index, params in enumerate(iter(self.queue.get, None)):
            self.params = params
            self.quantify_peaks(params)
        self.results.put(None)

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
            left, right = mean-2*stdl, mean+2*stdr
            scans[peak_isotope][xic_peak_index] = set(rt_scan_map[(rt_scan_map.index >= left) & (rt_scan_map.index <= right)].values)
    return scans

def run_pyquant():
    from . import pyquant_parser
    # for some reason this is being undefined in windows
    import numpy as np
    args = pyquant_parser.parse_args()
    isotopologue_limit = args.isotopologue_limit
    isotopologue_limit = isotopologue_limit if isotopologue_limit else None
    labels_needed = args.labels_needed
    overlapping_mz = args.overlapping_labels
    threads = args.p
    skip = args.skip
    out = args.out
    html = args.html
    resume = args.resume
    calc_stats = not args.disable_stats
    msn_for_id = args.msn_id
    mass_accuracy_correction = args.no_mass_accuracy_correction
    raw_data_only = not (args.search_file or args.tsv)
    scans_to_select = set(args.scan if args.scan else [])
    msn_for_quant = args.msn_quant_from if args.msn_quant_from else msn_for_id-1
    if msn_for_quant == 0:
        msn_for_quant = 1
    reporter_mode = args.reporter_ion
    msn_ppm = args.msn_ppm
    if args.msn_rt_window:
        msn_rt_window = [tuple(map(float, i.split('-'))) for i in args.msn_rt_window]
    else:
        msn_rt_window = None
    ref_label = str(args.reference_label) if args.reference_label else None
    quant_method = args.quant_method
    if msn_for_quant == 1 and quant_method is None:
        quant_method = 'integrate'
    elif msn_for_quant > 1 and quant_method is None:
        quant_method = 'sum'

    if args.isobaric_tags:
        reporter_mode = True
        mass_accuracy_correction = False
        calc_stats = False
        msn_for_quant = 2
    if args.ms3:
        msn_for_quant = 3
        mass_accuracy_correction = False

    mrm_pair_info = pd.read_table(args.mrm_map) if args.mrm and args.mrm_map else None

    scan_filemap = {}
    found_scans = {}
    raw_files = {}
    mass_labels = {'Light': {0: set([])}} if not reporter_mode else {}

    name_mapping = {}

    if args.label_scheme:
        mass_labels = {}
        label_info = pd.read_table(args.label_scheme.name, sep='\t', header=None, dtype='str')
        try:
            label_info.columns = ['Label', 'AA', 'Mass', 'UserName']
            name_mapping = dict([(v['Label'],v['UserName']) for i,v in label_info.iterrows()])
        except ValueError:
            label_info.columns = ['Label', 'AA', 'Mass']
            name_mapping = dict([(v['Label'],v['Label']) for i,v in label_info.iterrows()])
        for group_name, group_info in label_info.groupby('Label'):
            masses = {}
            label_name = name_mapping.get(group_name, group_name)
            for mass, mass_aas in group_info.groupby('Mass'):
                mass_val = float(mass)
                mass_list = mass_aas['AA'].values.tolist()
                try:
                    masses[mass_val].add(mass_list)
                except KeyError:
                    masses[mass_val] = set(mass_list)
            mass_labels.update({label_name: masses})
    if args.label_method:
        mass_labels = config.LABEL_SCHEMES[args.label_method]

    sample = args.sample
    sys.stderr.write('Loading Scans:\n')

    # options determining modes to quantify
    all_msn = False # we just have a raw file
    ion_search = False # we have an ion we want to find

    if args.peptide_file:
        peptides = set((i.strip().upper() for i in args.peptide_file))
    elif args.peptide:
        peptides = set(args.peptide) if isinstance(args.peptide, list) else set([args.peptide])
    else:
        peptides = None

    if peptides:
        peptides = set(map(lambda x: x.upper(), peptides))

    input_found = None
    if args.search_file:
        results = GuessIterator(args.search_file.name, full=True, store=False, peptide=peptides)
        input_found = 'ms'
    elif args.tsv:
        results = pd.read_table(args.tsv, sep='\t')
        input_found = 'tsv'

    if args.search_file:
        source_file = args.search_file.name
    elif args.tsv:
        source_file = args.tsv.name
    elif args.scan_file:
        source_file = args.scan_file[0].name

    if args.scan_file:
        nfunc = lambda i: (os.path.splitext(os.path.split(i.name)[1])[0], os.path.abspath(i.name)) if hasattr(i, 'name') else (os.path.splitext(os.path.split(i)[1])[0], os.path.abspath(i))
        scan_filemap = dict([nfunc(i) for i in args.scan_file])
    else:
        if args.scan_file_dir:
            raw_file = args.scan_file_dir
        else:
            raw_file = os.path.abspath(os.path.split(source_file)[0])
        if os.path.isdir(raw_file):
            scan_filemap = dict([(os.path.splitext(i)[0], os.path.abspath(os.path.join(raw_file, i))) for i in os.listdir(raw_file) if i.lower().endswith('mzml')])
        else:
            scan_filemap[os.path.splitext(os.path.split(raw_file)[1])[0]] = os.path.abspath(raw_file)

    if input_found == 'tsv':
        if args.maxquant:
            peptide_col = "Sequence"
            precursor_col = "m/z"
            rt_col = 'Retention time'
            charge_col = 'Charge'
            file_col = 'Raw file'
            if 'evidence' in source_file:
                scan_col = "MS/MS Scan Number"
                label_col = 'Labeling State'
            elif 'ms2' in source_file:
                scan_col = "Scan number"
                label_col = None
        else:
            peptide_col = args.peptide_col
            scan_col = args.scan_col
            precursor_col = args.mz
            rt_col = args.rt
            charge_col = args.charge
            file_col = args.source
            label_col = args.label
        for index, row in enumerate(results.iterrows()):
            if index%1000 == 0:
                sys.stderr.write('.')
            row_index, i = row
            peptide = i[peptide_col].strip() if peptide_col in i else ''
            if peptides and not any([j.lower() == peptide.lower() for j in peptides]):
                continue
            if not peptides and (sample != 1.0 and random.random() > sample):
                continue
            specId = str(i[scan_col])
            if scans_to_select and str(specId) not in scans_to_select:
                continue
            fname = i[file_col] if file_col in i else raw_file
            if fname not in scan_filemap:
                fname = os.path.split(fname)[1]
                if fname not in scan_filemap:
                    if skip:
                        continue
                    sys.stderr.write('{0} not found in filemap. Filemap is {1}. If you wish to ignore this message, add --skip to your input arguments.'.format(fname, scan_filemap))
                    return 1
            charge = float(i[charge_col]) if charge_col in i else 1
            precursor_mass = i[precursor_col] if precursor_col in i else None
            rt_value = i[rt_col] if rt_col in i else None
            mass_key = (specId, fname, charge, precursor_mass)
            if mass_key in found_scans:
                continue
            #'id': id_Scan[id], 'theor_mass' -> id_scan[mass], 'peptide': idScan[peptide,],  'mod_peptide': idscan, 'rt': idscan,
            #
            d = {
                'file': fname, 'quant_scan': {}, 'id_scan': {
                'id': specId, 'mass': precursor_mass, 'peptide': peptide, 'rt': rt_value,
                'charge': charge, 'modifications': None, 'label': name_mapping.get(i[label_col]) if label_col in i else None
            }
            }
            found_scans[mass_key] = d
            if args.mva:
                for i in scan_filemap:
                    raw_files[i] = d
            else:
                try:
                    raw_files[i[file_col]].append(d)
                except:
                    raw_files[i[file_col]] = [d]
    elif input_found == 'ms':
        if not (args.label_scheme or args.label_method):
            mass_labels.update(results.getSILACLabels())
        replicate_file_mapper = {}
        for index, scan in enumerate(results.getScans(modifications=False, fdr=True)):
            if index%1000 == 0:
                sys.stderr.write('.')
            if scan is None:
                continue
            peptide = scan.peptide
            # if peptide.lower() != 'AGkPVIcATQMLESmIk'.lower():
            #     continue
            if peptides and peptide.upper() not in peptides:
                continue
            if not peptides and (sample != 1.0 and random.random() > sample):
                continue
            specId = scan.id
            if scans_to_select and str(specId) not in scans_to_select:
                continue
            fname = scan.file
            mass_key = (fname, specId, peptide, scan.mass)
            if mass_key in found_scans:
                continue
            d = {
                    'file': fname, 'quant_scan': {}, 'id_scan': {
                    'id': specId, 'theor_mass': scan.getTheorMass(), 'peptide': peptide, 'mod_peptide': scan.modifiedPeptide, 'rt': scan.rt,
                    'charge': scan.charge, 'modifications': scan.getModifications(), 'mass': float(scan.mass), 'accession': getattr(scan, 'acc', None),
                }
            }
            found_scans[mass_key] = d#.add(mass_key)
            fname = os.path.splitext(fname)[0]
            if args.mva:
                # find the most similar name, add in a setting for this
                import difflib
                if fname in replicate_file_mapper:
                    fname = replicate_file_mapper[fname]
                    if fname is None:
                        continue
                else:
                    s = difflib.SequenceMatcher(None, fname)
                    seq_matches = []
                    for i in scan_filemap:
                        s.set_seq2(i)
                        seq_matches.append((s.ratio(), i))
                    seq_matches.sort(key=operator.itemgetter(0), reverse=True)
                    if False:#len(fname)-len(fname)*seq_matches[0][0] > 1:
                        replicate_file_mapper[fname] = None
                        continue
                    else:
                        replicate_file_mapper[fname] = seq_matches[0][1]
                        fname = seq_matches[0][1]
            if fname not in scan_filemap:
                fname = os.path.split(fname)[1]
                if fname not in scan_filemap:
                    if skip:
                        continue
                    sys.stderr.write('{0} not found in filemap. Filemap is {1}.'.format(fname, scan_filemap))
                    return 1
            try:
                raw_files[fname].append(d)
            except KeyError:
                raw_files[fname] = [d]
            del scan

    labels = mass_labels.keys()
    RESULT_ORDER = [
        ('peptide', 'Peptide'),
        ('modifications', 'Modifications'),
        ('accession', 'Accession'),
        ('charge', 'Charge'),
        ('ms1', 'MS{} Spectrum ID'.format(msn_for_quant)),
    ]
    if msn_for_quant != msn_for_id:
        RESULT_ORDER.extend([
            ('scan', 'MS{} Spectrum ID'.format(msn_for_id)),
        ])
    RESULT_ORDER.extend([
        ('rt', 'Retention Time'),
    ])

    PEAK_REPORTING = []
    if labels:
        iterator = [sorted(labels)[0]] if args.merge_labels else labels
        for silac_label in iterator:
            RESULT_ORDER.extend([
                ('{}_precursor'.format(silac_label), '{} Precursor'.format(silac_label)),
            ])
            if msn_for_quant == 1:
                RESULT_ORDER.extend([
                    ('{}_calibrated_precursor'.format(silac_label), '{} Calibrated Precursor'.format(silac_label)),
                    ('{}_isotopes'.format(silac_label), '{} Isotopes Found'.format(silac_label)),
                ])
            RESULT_ORDER.extend([
                ('{}_intensity'.format(silac_label), '{} Intensity'.format(silac_label)),
            ])
            if msn_for_quant == 1:
                RESULT_ORDER.extend([
                    ('{}_residual'.format(silac_label), '{} Residual'.format(silac_label)),
                ])

            if not reporter_mode:
                PEAK_REPORTING.extend([
                    ('auc', '{} Peak Area'.format(silac_label)),
                    ('amp', '{} Peak Max'.format(silac_label)),
                    ('mean', '{} Peak Center'.format(silac_label)),
                    ('peak_width', '{} RT Width'.format(silac_label)),
                    ('mean_diff', '{} Mean Offset'.format(silac_label)),
                    ('snr', '{} SNR'.format(silac_label)),
                    ('sbr', '{} SBR'.format(silac_label)),
                    ('sdr', '{} Density'.format(silac_label)),
                    # ('residual', '{} Residual'.format(silac_label)),
                ])
            if args.peaks_n == 1:
                RESULT_ORDER.extend(PEAK_REPORTING)
                PEAK_REPORTING = []
            if not args.merge_labels and not args.no_ratios:
                for silac_label2 in labels:
                    if silac_label != silac_label2 and (ref_label is None or ref_label.lower() == silac_label2.lower()):
                        RESULT_ORDER.extend([('{}_{}_ratio'.format(silac_label, silac_label2), '{}/{}'.format(silac_label, silac_label2)),
                                             ])
                        if calc_stats:
                            RESULT_ORDER.extend([('{}_{}_confidence'.format(silac_label, silac_label2), '{}/{} Confidence'.format(silac_label, silac_label2)),
                                                 ])

    if scan_filemap and raw_data_only:
        # pop the peptide/mods from result_order
        to_pop = []
        to_pop_keys = {'peptide', 'modifications', 'accession'}
        for i,v in enumerate(RESULT_ORDER):
            if v[0] in to_pop_keys:
                to_pop.append(i)
        for i in sorted(to_pop, reverse=True):
            del RESULT_ORDER[i]
        # determine if we want to do ms1 ion detection, ms2 ion detection, all ms2 of each file
        if args.msn_ion or args.msn_peaklist:
            RESULT_ORDER.extend([('ions_found', 'Ions Found')])
            ion_search = True
            ions_selected = args.msn_ion if args.msn_ion else []#
            if args.msn_peaklist:
                ion_table = pd.read_table(args.msn_peaklist)

            # [float(i.strip()) for i in args.msn_peaklist if i]
            d = {'ions': ions_selected, 'rt_info': args.msn_ion_rt}
            for i in scan_filemap:
                raw_files[i] = d
        else:
            all_msn = True
            for i in scan_filemap:
                raw_files[i] = [1]
    if not scan_filemap and input_found is None:
        sys.stderr.write('No valid input entered. PyQuant requires at least a raw file or a processed dataset.')
        return 1
    sys.stderr.write('\nScans loaded.\n')

    pq_dir = os.path.split(__file__)[0]

    pyquant_html_file = os.path.join(pq_dir, 'static', 'pyquant_output.html')

    workers = []
    completed = 0
    sys.stderr.write('Beginning quantification.\n')
    scan_count = len(found_scans)
    headers = ['Raw File']+[i[1] for i in RESULT_ORDER]
    if resume and os.path.exists(out):
        if not out:
            sys.stderr.write('You may only resume runs with a file output.\n')
            return -1
        out = open(out, 'a+')
        out_path = out.name
    else:
        if out:
            out = open(out, 'w+')
            out_path = out.name
        else:
            out = sys.stdout
            out_path = source_file
        out.write('{0}\n'.format('\t'.join(headers)))

    if html:
        def table_rows(html_list, res=None):
            # each item is a string like a\tb\tc
            if html_list:
                d = html_list.pop(0)
            else:
                return res
            l = d['table']
            html_extra = d.get('html', {})
            keys = d['keys']
            if res is None:
                res = '<tr>'
            out = ['<td></td>']# for graph controls
            for col_index, (i,v) in enumerate(zip(l.split('\t'), keys)):
                out.append('<td>{0}</td>'.format(i))
            res += '\n'.join(out)+'</tr>'
            return table_rows(html_list, res=res), html_extra

        if resume:
            html_out = open('{0}.html'.format(out_path), 'a')
        else:
            html_out = open('{0}.html'.format(out_path), 'w')
            template = []
            for i in open(pyquant_html_file, 'r'):
                if 'HTML BREAK' in i:
                    break
                template.append(i)
            html_template = Template(''.join(template))
            html_out.write(html_template.substitute({'title': source_file, 'table_header': '\n'.join(['<th>{0}</th>'.format(i) for i in ['Controls', 'Raw File']+[i[1] for i in RESULT_ORDER]])}))

    skip_map = set([])
    if resume:
        resume_name = '{}.tmp'.format(out.name)
        if os.path.exists(resume_name):
            with open(resume_name, 'r') as temp_file:
                for index, entry in enumerate(temp_file):
                    info = json.loads(entry)['res_list']
                    # key is filename, peptide, charge, target scan id, modifications
                    key = tuple(map(str, (info[0], info[1], info[3], info[5], info[2])))
                    skip_map.add(key)
            temp_file = open(resume_name, 'a')
        else:
            temp_file = open(resume_name, 'w')
    else:
        temp_file = open('{}.tmp'.format(out.name), 'w')

    silac_shifts = {}
    for silac_label, silac_masses in mass_labels.items():
        for mass, aas in six.iteritems(silac_masses):
            try:
                silac_shifts[mass] |= aas
            except:
                silac_shifts[mass] = aas

    for filename in raw_files.keys():
        raw_scans = raw_files[filename]
        filepath = scan_filemap[filename]
        if not len(raw_scans):
            continue
        in_queue = Queue()
        result_queue = Queue()
        reader_in = Queue()
        reader_outs = {'main': Queue()}
        for i in xrange(threads):
            reader_outs[i] = Queue()

        msn_map = []
        scan_rt_map = {}
        msn_rt_map = {}
        scan_charge_map = {}

        raw = GuessIterator(filepath, full=False, store=False)
        sys.stderr.write('Processing {}.\n'.format(filepath))

        # params in case we are doing ion search or replicate analysis
        ion_tolerance = args.precursor_ppm/1e6 if args.mva else args.msn_ppm/1e6
        ion_search_list = []
        replicate_search_list = defaultdict(list)
        scans_to_fetch = []

        # figure out the splines for mass accuracy correction
        calc_spline = not mass_accuracy_correction and not raw_data_only and not args.neucode
        spline_x = []
        spline_y = []
        spline = None

        scan_info_map = defaultdict(dict)

        for index, scan in enumerate(raw):
            if index % 100 == 0:
                sys.stderr.write('.')
            if scan is None:
                continue
            scan_id = scan.id
            msn_map.append((scan.ms_level if not args.mrm else scan.mass, scan_id))
            rt = scan.rt
            if scan.ms_level == msn_for_quant:
                msn_rt_map[scan_id] = int(scan.title) if args.mrm else rt
            scan_rt_map[scan_id] = rt
            scan_info_map[scan_id]['parent'] = scan.parent
            scan_info_map[scan_id]['msn'] = scan.ms_level
            scan_info_map[scan_id]['precursor'] = scan.mass
            scan_charge_map[scan_id] = scan.charge
            if msn_rt_window and not any([(i[0] < float(rt) < i[1]) for i in msn_rt_window]):
                continue
            if scan.parent:
                try:
                    scan_info_map[scan.parent]['children'].add(scan_id)
                except KeyError:
                    scan_info_map[scan.parent]['children'] = set([scan_id])
            if not scans_to_select or str(scan_id) in scans_to_select:
                if ion_search:
                    if scan.ms_level == msn_for_id:
                        scans_to_fetch.append(scan_id)
                elif all_msn:
                    # we are quantifying all msn spectra of a given type
                    if msn_for_id == scan.ms_level:
                        # find the closest scan to this, which will be the parent scan
                        spectra_to_quant = find_prior_scan(msn_map, scan_id, ms_level=msn_for_quant) if msn_for_quant != msn_for_id else scan_id
                        d = {
                            'quant_scan': {'id': spectra_to_quant},
                            'id_scan': {'id': scan_id, 'rt': scan.rt, 'charge': scan.charge, 'mass': float(scan.mass), 'product_ion': float(scan.product_ion) if args.mrm else None},
                        }
                        ion_search_list.append((spectra_to_quant, d))
                elif args.mva:
                    if scan.ms_level == msn_for_quant:
                        scans_to_fetch.append(scan_id)
            if not raw_data_only and calc_spline:
                if hasattr(scan, 'theor_mass'):
                    theor_mass = scan.getTheorMass()
                    observed_mass = scan.mass
                    mass_error = (theor_mass-observed_mass)/theor_mass*1e6
                    spline_x.append(observed_mass)
                    spline_y.append(mass_error)
            del scan

        if calc_spline and len(spline_x):
            spline_df = pd.DataFrame(zip(spline_x, spline_y), columns=['Observed', 'Error'])
            spline_df = spline_df[(spline_df['Error']<25) & (spline_df['Error']>-25)].dropna()
            spline_df.sort('Observed', inplace=True)
            spline_df.drop_duplicates('Observed', inplace=True)
            if len(spline_df) > 10:
                spline = UnivariateSpline(spline_df['Observed'].astype(float).values, spline_df['Error'].astype(float).values, s=1e6)
            else:
                spline = None

        reader = Reader(reader_in, reader_outs, raw_file=filepath, spline=spline, rt_window=msn_rt_window)
        reader.start()
        rep_map = defaultdict(set)
        if ion_search or args.mva:
            ions = [i['id_scan'].get('theor_mass', i['id_scan']['mass']) for i in raw_scans] if args.mva else raw_scans['ions']
            rt_info = raw_scans.get('rt_info') if not args.mva else None
            last_scan_ions = defaultdict(set)
            for scan_id in scans_to_fetch:
                this_scan_ions = defaultdict(set)
                reader_in.put((0, scan_id, None, None))
                scan = reader_outs[0].get()
                if scan is None:
                    continue
                if args.msn_all_scans:
                    for ion_index, ion in enumerate(ions):
                        d = {
                            'quant_scan': {'id': scan_id, 'scans': scans_to_fetch},
                            'id_scan': {
                                'id': scan_id, 'theor_mass': ion, 'rt': rt_info[ion_index] if rt_info else scan['rt'],
                                'charge': 1, 'mass': float(ion), 'ions_found': ion,
                            },
                        }
                        ion_search_list.append((scan_id, d))
                    break
                scan_mzs = scan['vals']
                mz_vals = scan_mzs[scan_mzs[:, 1] > 0][:, 1]
                scan_mzs = scan_mzs[scan_mzs[:, 1] > 0][:, 0]
                if not np.any(scan_mzs):
                    continue
                mass, charge, rt = scan['mass'], scan['charge'], scan['rt']
                ions_found = []
                added = set([])
                for ion_index, (ion, nearest_mz_index) in enumerate(zip(ions, peaks.find_nearest_indices(scan_mzs, ions))):
                    nearest_mz = scan_mzs[nearest_mz_index]
                    if peaks.get_ppm(ion, nearest_mz) < ion_tolerance:
                        if args.mva:
                            d = raw_scans[ion_index]
                            scan_rt = d['id_scan']['rt']
                            if scan_rt-args.rt_window < rt < scan_rt+args.rt_window:
                                ions_found.append({'ion': ion, 'nearest_mz': nearest_mz, 'charge': d['id_scan']['charge'], 'scan_info': d})
                        else:
                            ion_rt = rt_info[ion_index] if rt_info else None
                            if ion_rt is None or (rt-args.rt_window < ion_rt < rt+args.rt_window):
                                ions_found.append({'ion': ion, 'nearest_mz': nearest_mz, 'rt': ion_rt})
                if ions_found:
                    # we have two options here. If we are quantifying a preceeding scan or the ion itself per scan
                    isotope_ppm = args.isotope_ppm/1e6
                    if msn_for_quant == msn_for_id or args.mva:
                        for ion_dict in ions_found:
                            ion, nearest_mz = ion_dict['ion'], ion_dict['nearest_mz']
                            ion_found = '{}({})'.format(ion, nearest_mz)
                            spectra_to_quant = scan_id
                            # we are quantifying the ion itself
                            if charge == 0 or args.mva:
                                # see if we can figure out the charge state
                                charge_states = []
                                for i in xrange(1, 5):
                                    charge_peaks_found = 0
                                    peak_height = 0
                                    for j in xrange(1, 3):
                                        next_peak = ion+peaks.NEUTRON/float(i)*float(j)
                                        closest_mz = peaks.find_nearest_index(scan_mzs, next_peak)
                                        if peaks.get_ppm(next_peak, scan_mzs[closest_mz]) < isotope_ppm*1.5:
                                            charge_peaks_found += 1
                                            peak_height += mz_vals[closest_mz]
                                    charge_states.append((charge_peaks_found, i, peak_height))
                                # print int(ion_dict['charge']), charge_states
                                # print int(ion_dict['charge']), charge_states
                                charge_states = sorted(charge_states, key=operator.itemgetter(0, 2), reverse=True)
                                if args.mva and int(ion_dict['charge']) not in [i[0] for i in charge_states]:
                                    continue
                                elif args.mva:
                                    charge_to_use = ion_dict['charge']
                                elif charge_states:
                                    if charge_states[0][1] == 1:
                                        charge_to_use = charge_states[1][1] if charge_states[1][2] != 0 else 1
                                    else:
                                        charge_to_use = charge_states[0][1]
                                else:
                                    charge_to_use = 1
                                if args.mva:
                                    rep_key = (ion_dict['scan_info']['id_scan']['rt'], ion_dict['scan_info']['id_scan']['mass'], charge_to_use)
                                    rep_map[rep_key].add(rt)
                            else:
                                charge_to_use = charge
                            this_scan_ions[ion].add(charge_to_use)
                            if charge_to_use in last_scan_ions[ion]:
                                continue
                            last_scan_ions[ion].add(charge_to_use)
                            if args.mva:
                                d = copy.deepcopy(ion_dict['scan_info'])
                                d['id_scan']['id'] = scan_id
                                theo_mass = d['id_scan'].get('theor_mass')
                                if theo_mass:
                                    d['id_scan']['mass'] = theo_mass
                                spectra_to_quant = find_prior_scan(msn_map, scan_id, ms_level=msn_for_quant)
                                d['quant_scan']['id'] = spectra_to_quant
                                d['replicate_scan_rt'] = rt
                            else:
                                d = {
                                    'quant_scan': {'id': scan_id},
                                    'id_scan': {
                                        'id': scan_id, 'theor_mass': ion, 'rt': rt if ion_rt is None else ion_rt,
                                        'charge': charge_to_use, 'mass': float(nearest_mz), 'ions_found': ion_found,
                                    },
                                }
                            key = (scan_id, d['id_scan']['theor_mass'], charge_to_use)
                            if key in added:
                                continue
                            added.add(key)
                            if args.mva:
                                replicate_search_list[(d['id_scan']['rt'], ion_dict['ion'])].append((spectra_to_quant, d))
                            else:
                                ion_search_list.append((spectra_to_quant, d))
                            # print 'adding', ion, nearest_mz, d
                    else:
                        # we are identifying the ion in a particular scan, and quantifying a preceeding scan
                        # find the closest scan to this, which will be the parent scan
                        spectra_to_quant = find_prior_scan(msn_map, scan_id, ms_level=msn_for_quant)
                        d = {
                            'quant_scan': {'id': spectra_to_quant},
                            'id_scan': {
                                'id': scan_id, 'rt': rt, 'charge': charge,
                                'mass': float(mass), 'ions_found': ';'.join(map(lambda x: '{}({})'.format(ion, nearest_mz), ions_found))
                            },
                        }
                        ion_search_list.append((spectra_to_quant, d))
                to_remove = []
                for ion, charges in last_scan_ions.items():
                    for charge in charges:
                        if charge not in this_scan_ions[ion]:
                            to_remove.append((ion, charge))
                for ion, charge in to_remove:
                    last_scan_ions[ion].discard(charge)
                del scan

        if args.mva:
            x = []
            y = []
            for i,v in six.iteritems(rep_map):
                for j in v:
                    x.append(i[0])
                    y.append(j)
            from sklearn.linear_model import LinearRegression
            rep_mapper = LinearRegression()
            try:
                rep_mapper.fit(np.array(x).reshape(len(x), 1), y)
            except ValueError:
                rep_mapper = None

        if ion_search or all_msn:
            raw_scans = [i[1] for i in sorted(ion_search_list, key=operator.itemgetter(0))]
        if args.mva:
            raw_scans = []
            for i in replicate_search_list:
                ion_rt, ion = i
                best_scan = sorted([(np.abs((rep_mapper.predict(j[1]['replicate_scan_rt']) if rep_mapper else j[1]['replicate_scan_rt'])-ion_rt), j[1]) for j in replicate_search_list[i]], key=operator.itemgetter(0))[0][1]
                # import pdb; pdb.set_trace();
                raw_scans.append(best_scan)

        for i in xrange(threads):
            worker = Worker(queue=in_queue, results=result_queue, raw_name=filepath, mass_labels=mass_labels,
                            debug=args.debug, html=html, mono=not args.spread, precursor_ppm=args.precursor_ppm,
                            isotope_ppm=args.isotope_ppm, isotope_ppms=None, msn_rt_map=msn_rt_map, reporter_mode=reporter_mode,
                            reader_in=reader_in, reader_out=reader_outs[i], thread=i, quant_method=quant_method,
                            spline=spline, isotopologue_limit=isotopologue_limit, labels_needed=labels_needed,
                            quant_msn_map=[i for i in msn_map if i[0] == msn_for_quant] if not args.mrm else msn_map,
                            overlapping_mz=overlapping_mz, min_resolution=args.min_resolution, min_scans=args.min_scans,
                            mrm_pair_info=mrm_pair_info, mrm=args.mrm, peak_cutoff=args.peak_cutoff, replicate=args.mva,
                            ref_label=ref_label, max_peaks=args.max_peaks, parser_args=args)
            workers.append(worker)
            worker.start()

        # TODO:
        # combine information from scans (where for instance, we have fragmented both the heavy/light
        # peptides -- we want to use those masses before calculating where it should be). This may not
        # be possible for all types of input though, figure this out.
        quant_map = {}
        scans_to_submit = []

        # this is to fix the header at the end to include peak information if we have multiple peaks
        most_peaks_found = 0

        lowest_label = min([j for i,v in mass_labels.items() for j in v])
        mrm_added = set([])
        exclusion_masses = mrm_pair_info.loc[:,[i for i in mrm_pair_info.columns if i.lower() not in ('light', 'retention time')]].values.flatten() if args.mrm else set([])
        for scan_index, v in enumerate(raw_scans):
            target_scan = v['id_scan']
            quant_scan = v['quant_scan']
            scanId = target_scan['id']
            scan_mass = target_scan.get('mass')
            if args.mrm:
                if scan_mass in mrm_added:
                    continue
                mrm_added.add(scan_mass)
                if scan_mass in exclusion_masses:
                    continue

            if quant_scan.get('id') is None:
                # we will hit this in a normal proteomic run
                # figure out the ms-1 from the ms level we are at
                if msn_for_quant > msn_for_id:
                    children = scan_info_map[scanId]['children']
                    quant_scan['scans'] = []
                    scan_to_quant = None
                    for child in children:
                        child_scan_info = scan_info_map[child]
                        scan_to_quant_ms = child_scan_info['msn']
                        if scan_to_quant_ms == msn_for_quant:
                            if scan_to_quant is None:
                                scan_to_quant = child
                            quant_scan['scans'].append(child)
                else:
                    scan_info = scan_info_map[scanId]
                    scan_to_quant = scan_info['parent']
                    try:
                        scan_to_quant_ms = scan_info[scan_to_quant]['msn']
                        while scan_to_quant and scan_to_quant != msn_for_quant:
                            scan_info = scan_info_map[scanId]
                            scan_to_quant = scan_info['parent']
                            scan_to_quant_ms = scan_info[scan_to_quant]['msn']
                    except KeyError:
                        scan_to_quant = None
                if scan_to_quant is not None and scan_to_quant_ms == msn_for_quant:
                    msn_to_quant = scan_to_quant
                else:
                    msn_to_quant = find_prior_scan(msn_map, scanId, ms_level=msn_for_quant)
                quant_scan['id'] = msn_to_quant

            rt = target_scan.get('rt', scan_rt_map.get(scanId))
            if rt is None:
                rt = float(msn_rt_map[msn_to_quant])
                target_scan['rt'] = rt

            if args.mva and rep_mapper is not None:
                target_scan['rt'] = rep_mapper.predict(float(target_scan['rt']))[0]

            mods = target_scan.get('modifications')
            charge = target_scan.get('charge')
            if charge is None or charge == 0:
                charge = int(scan_charge_map.get(scanId, 0))
            if charge == 0:
                continue
            charge = int(charge)

            if msn_for_quant != 1:
                target_scan['theor_mass'] = lowest_label
                target_scan['precursor'] = lowest_label
            else:
                mass_shift = 0

                if mods is not None:
                    shift = 0
                    for mod in filter(lambda x: x, mods.split('|')):
                        aa, pos, mass, _ = mod.split(',', 3)
                        mass = float('{0:0.5f}'.format(float(mass)))
                        if aa in silac_shifts.get(mass, {}):
                            shift += mass
                    mass_shift += (float(shift)/float(charge))
                else:
                    # assume we are the light version, include all the labels we are looking for here
                    pass

                target_scan['theor_mass'] = target_scan.get('theor_mass', target_scan.get('mass'))-mass_shift
                target_scan['precursor'] = target_scan['mass']-mass_shift if not args.neucode else target_scan['theor_mass']
            # key is filename, peptide, charge, target scan id, modifications
            key = (filename, target_scan.get('peptide', ''), target_scan.get('charge'), target_scan.get('id'), target_scan.get('modifications'),)
            if resume:
                if tuple(map(str, key)) in skip_map:
                    completed += 1
                    continue
            params = {'scan_info': v}
            scans_to_submit.append((target_scan['rt'], params))

        # sort by RT so we can minimize our memory footprint by throwing away scans we no longer need
        scans_to_submit.sort(key=operator.itemgetter(0))
        if ion_search or all_msn or args.mva:
            scan_count = len(scans_to_submit)
        for i in scans_to_submit:
            in_queue.put(i[1])

        sys.stderr.write('{0} processed and placed into queue.\n'.format(filename))

        # kill the workers
        [in_queue.put(None) for i in xrange(threads)]
        RESULT_DICT = {i[0]: 'NA' for i in RESULT_ORDER}
        scans_to_export = set([])
        export_mapping = defaultdict(set)
        key_map = msn_rt_map.keys()
        rt_scan_map = pd.Series(key_map, index=[msn_rt_map[i] for i in key_map])
        rt_scan_map.sort_index(inplace=True)
        while workers or result is not None:
            try:
                result = result_queue.get(timeout=0.1)
            except Empty:
                # kill expired workers
                result = None
                to_del = []
                for i, v in enumerate(workers):
                    if not v.is_alive():
                        v.terminate()
                        exit_code = v.exitcode
                        if exit_code in CRASH_SIGNALS:
                            print('thread has been killed using params {}'.format(v.params))
                        to_del.append({'worker_index': i, 'thread_id': v.thread, 'exitcode': exit_code})
                workers_to_add = []
                for worker_dict in sorted(to_del, key=operator.itemgetter('worker_index'), reverse=True):
                    worker_index = worker_dict['worker_index']
                    if worker_dict['exitcode'] in CRASH_SIGNALS:
                        thread_index = worker_dict['thread_id']
                        worker = Worker(queue=in_queue, results=result_queue, raw_name=filepath, mass_labels=mass_labels,
                                debug=args.debug, html=html, mono=not args.spread, precursor_ppm=args.precursor_ppm,
                                isotope_ppm=args.isotope_ppm, isotope_ppms=None, msn_rt_map=msn_rt_map, reporter_mode=reporter_mode,
                                reader_in=reader_in, reader_out=reader_outs[thread_index], thread=thread_index, quant_method=quant_method,
                                spline=spline, isotopologue_limit=isotopologue_limit, labels_needed=labels_needed,
                                quant_msn_map=[i for i in msn_map if i[0] == msn_for_quant] if not args.mrm else msn_map,
                                overlapping_mz=overlapping_mz, min_resolution=args.min_resolution, min_scans=args.min_scans,
                                mrm_pair_info=mrm_pair_info, mrm=args.mrm, peak_cutoff=args.peak_cutoff, replicate=args.mva,
                                ref_label=ref_label, max_peaks=args.max_peaks, parser_args=args)
                        workers_to_add.append(worker)
                        worker.start()
                    del workers[worker_index]
                workers += workers_to_add
            if result is not None:
                completed += 1
                if completed % 10 == 0:
                    sys.stderr.write('\r{0:2.2f}% Completed'.format(completed/scan_count*100))
                    sys.stderr.flush()
                res_dict = copy.deepcopy(RESULT_DICT)
                for i in RESULT_ORDER:
                    res_dict[i[0]] = result.get(i[0], 'NA')
                peak_report = []
                for label_name in labels:
                    peaks_found = result.get('{}_peaks'.format(label_name), {})
                    if args.export_mzml:
                        from . import PER_FILE, PER_ID, PER_PEAK
                        scans = get_scans_under_peaks(rt_scan_map, peaks_found)
                        flattened_scans = set([l for i,v in scans.items() for j,k in v.items() for l in k])
                        scans_to_export |= flattened_scans
                        if args.export_mode == PER_FILE:
                            export_mapping['{}_{}.mzML'.format(out_path, filename)] |= flattened_scans
                    if len(peaks_found) > most_peaks_found:
                        most_peaks_found = len(peaks_found)
                    for isotope_index, isotope_peaks in six.iteritems(peaks_found):
                        if args.export_mzml and args.export_mode == PER_ID:
                            export_mapping['{out}_{raw}_{ms1}_{precursor}.mzML'.format(**{
                                    'out': out_path,
                                    'raw': filename,
                                    'precursor': result.get('{}_precursor'.format(label_name)),
                                    'ms1': result.get('ms1')
                                })] |= set([l for i,v in scans[isotope_index].items() for l in v])
                        for xic_peak_index, xic_peak_info in six.iteritems(isotope_peaks):
                            if args.export_mzml and args.export_mode == PER_PEAK:
                                export_mapping['{out}_{raw}_{ms1}_{precursor}_{isotope}_{peak}.mzML'.format(**{
                                    'out': out_path,
                                    'raw': filename,
                                    'peak': xic_peak_index,
                                    'isotope': isotope_index,
                                    'precursor': result.get('{}_precursor'.format(label_name)),
                                    'ms1': result.get('ms1')
                                })] |= scans[isotope_index][xic_peak_index]
                            if args.peaks_n != 1:
                                peak_report.append(list(map(str, (xic_peak_info.get(i[0], 'NA') for i in PEAK_REPORTING))))
                            else:
                                for i in RESULT_ORDER:
                                    if i[0] in xic_peak_info:
                                        res_dict[i[0]] = xic_peak_info[i[0]]
                res_dict['filename'] = filename
                res_dict['peak_report'] = peak_report
                # This is the tsv output we provide
                res_list = [filename]+[res_dict.get(i[0], 'NA') for i in RESULT_ORDER]+['\t'.join(i) for i in peak_report]
                res = '{0}\n'.format('\t'.join(map(str, res_list)))
                out.write(res)
                out.flush()

                temp_file.write(json.dumps({'res_dict': res_dict, 'html': result.get('html', {})}))
                temp_file.write('\n')
                temp_file.flush()

        if scans_to_export:
            for export_filename, scans in six.iteritems(export_mapping):
                with open(export_filename, 'w') as o:
                    raw.writeScans(handle=o, scans=scans)

        reader_in.put(None)


        del msn_map
        del scan_rt_map
        del raw

    out.flush()
    out.close()
    # fix the header if we need to. We reopen the file because Windows doesn't like it when we read on a file with 'w+'
    if args.peaks_n != 1:
        tmp_file = '{}_ot'.format(out_path)
        with open(out.name, 'r') as out:
            with open(tmp_file, 'w') as o:
                new_header = out.readline().strip().split('\t')
                for peak_num in xrange(most_peaks_found):
                    new_header.extend(['Peak {} {}'.format(peak_num, i[1]) for i in PEAK_REPORTING])
                o.write('{}\n'.format('\t'.join(new_header)))
                for row in out:
                    o.write(row)
        import shutil
        shutil.move(tmp_file, out_path)


    temp_file.close()
    df_data = []
    html_data = []
    peak_data = []
    for j in open(temp_file.name, 'r'):
        result = json.loads(j if isinstance(j, six.text_type) else j.decode('utf-8'))
        res_dict = result['res_dict']
        res_list = [res_dict['filename']]+[res_dict.get(i[0], 'NA') for i in RESULT_ORDER]
        peak_data.append(res_dict['peak_report'])
        df_data.append(res_list)
        html_data.append(result['html'])
    data = pd.DataFrame.from_records(df_data, columns=[i for i in headers if i != 'Confidence'])
    header_mapping = []
    order_names = [i[1] for i in RESULT_ORDER]
    for i in data.columns:
        try:
            header_mapping.append(RESULT_ORDER[order_names.index(i)][0])
        except ValueError:
            header_mapping.append(False)
    if calc_stats and six.PY2:
        from scipy import stats
        import pickle
        import numpy as np
        classifier = pickle.load(open(os.path.join(pq_dir, 'static', 'classifier.pickle'), 'rb'))
        data = data.replace('NA', np.nan)
        for silac_label1 in mass_labels.keys():
            label1_log = 'L{}'.format(silac_label1)
            label1_logp = 'L{}_p'.format(silac_label1)
            label1_int = '{} Intensity'.format(silac_label1)
            label1_pint = '{} Peak Intensity'.format(silac_label1)
            label1_hif = '{} Isotopes Found'.format(silac_label1)
            label1_hifp = '{} Isotopes Found p'.format(silac_label1)
            for silac_label2 in mass_labels.keys():
                if silac_label1 == silac_label2:
                    continue
                try:
                    label2_log = 'L{}'.format(silac_label2)
                    label2_logp = 'L{}_p'.format(silac_label2)
                    label2_int = '{} Intensity'.format(silac_label2)
                    label2_pint = '{} Peak Intensity'.format(silac_label2)
                    label2_hif = '{} Isotopes Found'.format(silac_label2)
                    label2_hifp = '{} Isotopes Found p'.format(silac_label2)

                    mixed = '{}/{}'.format(silac_label1, silac_label2)
                    mixed_p = '{}/{}_p'.format(silac_label1, silac_label2)
                    mixed_mean = '{}_Mean_Diff'.format(mixed)
                    mixed_mean_p = '{}_Mean_Diff_p'.format(mixed)
                    mixed_rt_diff = '{}_RT_Diff'.format(mixed)
                    mixed_rt_diff_p = '{}_p'.format(mixed_rt_diff)
                    mixed_isotope_diff = '{}_Isotope_Diff'.format(mixed)
                    mixed_isotope_diff_p = '{}_Isotope_Diff_p'.format(mixed)

                    data[label1_log] = np.log(data[label1_int].astype(float)+1)
                    data[label1_logp] = stats.norm.cdf((data[label1_log] - data[data[label1_log]>0][label1_log].mean())/data[data[label1_log]>0][label1_log].std(ddof=0))
                    data[label2_log] = np.log(data[label2_int].astype(float)+1)
                    data[label2_logp] = stats.norm.cdf((data[label2_log] - data[data[label2_log]>0][label2_log].mean())/data[data[label2_log]>0][label2_log].std(ddof=0))

                    nz = data[(data[label2_log] > 0) & (data[label1_log] > 0)]
                    mu = pd.Series(np.ravel(nz.loc[:,(label2_log, label1_log)])).mean()
                    std = pd.Series(np.ravel(nz.loc[:,(label2_log, label1_log)])).std()

                    data[mixed_p] = stats.norm.cdf((data.loc[:,(label2_log, label1_log)].mean(axis=1)-mu)/std)
                    data[mixed_rt_diff] = np.log2(np.abs(data['{} RT Width'.format(silac_label2)].astype(float)-data['{} RT Width'.format(silac_label1)].astype(float)))
                    data[mixed_mean] = np.abs(data['{} Mean Offset'.format(silac_label1)].astype(float)-data['{} Mean Offset'.format(silac_label1)].astype(float))
                    data[mixed_rt_diff] = data[mixed_rt_diff].replace([np.inf, -np.inf], np.nan)
                    data[mixed_rt_diff_p] = stats.norm.cdf((data[mixed_rt_diff] - data[mixed_rt_diff].mean())/data[mixed_rt_diff].std(ddof=0))
                    data[mixed_mean_p] = stats.norm.cdf((data[mixed_mean] - data[mixed_mean].mean())/data[mixed_mean].std(ddof=0))

                    data[label2_hif] = data[label2_hif].astype(float)
                    data[label2_hifp] = np.log2(data[label2_hif]).replace([np.inf, -np.inf], np.nan)
                    data[label2_hifp] = stats.norm.cdf((data[label2_hifp]-data[label2_hifp].median())/data[label2_hifp].std())

                    data[label1_hif] = data[label1_hif].astype(float)
                    data[label1_hifp] = np.log2(data[label1_hif]).replace([np.inf, -np.inf], np.nan)
                    data[label1_hifp] = stats.norm.cdf((data[label1_hifp]-data[label1_hifp].median())/data[label2_hifp].std())

                    data[mixed_isotope_diff] = np.log2(data[label2_hif]/data[label1_hif]).replace([np.inf, -np.inf], np.nan)
                    data[mixed_isotope_diff_p] = stats.norm.cdf((data[mixed_isotope_diff] - data[mixed_isotope_diff].median())/data[mixed_isotope_diff].std(ddof=0))

                    # confidence assessment
                    mixed_confidence = '{}/{} Confidence'.format(silac_label1, silac_label2)

                    cols = []
                    for i in (silac_label1, silac_label2):
                        cols.extend(['{} {}'.format(i,j) for j in ['Intensity', 'Isotopes Found', 'Peak Area', 'SNR', 'Residual']])

                    fit_data = data.loc[:, cols]

                    # TODO: Fix this in hte model to be peak area
                    fit_data.rename(columns={'{} Peak Area'.format(silac_label1): label1_pint, '{} Peak Area'.format(silac_label2): label2_pint}, inplace=True)
                    fit_data.loc[:,(label2_int, label1_int, label2_pint, label1_pint)] = np.log2(fit_data.loc[:,(label2_int, label1_int, label2_pint, label1_pint)].astype(float))
                    from sklearn import preprocessing
                    from patsy import dmatrix
                    fit_data = fit_data.replace([np.inf, -np.inf], np.nan)
                    non_na_data = fit_data.dropna().index
                    fit_data.dropna(inplace=True)
                    fit_data.loc[:] = preprocessing.scale(fit_data)

                    formula = "{columns} + (Q('{silac_label1} Intensity')*Q('{silac_label1} Peak Intensity')*Q('{silac_label1} SNR'))**2+(Q('{silac_label2} SNR')*Q('{silac_label2} Intensity')*Q('{silac_label2} Peak Intensity'))**2".format(**{
                        'columns': "+".join(["Q('{}')".format(i) for i in fit_data.columns]),
                        'silac_label1': silac_label1,
                        'silac_label2': silac_label2,
                    })
                    fit_predictors = dmatrix(str(formula), fit_data)

                    # for a 'good' vs. 'bad' classifier, where 1 is good
                    conf_ass = classifier.predict_proba(fit_predictors)[:,1]*10
                    data.loc[non_na_data, mixed_confidence] = conf_ass

                    # conf_ass = classifier.predict(fit_data.values)
                    # left = conf_ass[conf_ass<0]
                    # ecdf = pd.Series(left).value_counts().sort_index().cumsum()*1./len(left)*10
                    # #ecdf = pd.Series(conf_ass).value_counts().sort_index(ascending=False).cumsum()*1./len(conf_ass)*10
                    # mapper = interp1d(ecdf.index.values, ecdf.values)
                    # data.loc[non_na_data[conf_ass<0], mixed_confidence] = mapper(left)
                    # right = conf_ass[conf_ass>=0]
                    # ecdf = pd.Series(right).value_counts().sort_index(ascending=False).cumsum()*1./len(right)*10
                    # #ecdf = pd.Series(conf_ass).value_counts().sort_index(ascending=False).cumsum()*1./len(conf_ass)*10
                    # mapper = interp1d(ecdf.index.values, ecdf.values)
                    # data.loc[non_na_data[conf_ass>=0], mixed_confidence] = mapper(right)
                except:
                    sys.stderr.write('Unable to calculate statistics for {}/{}.\n Traceback: {}'.format(silac_label1, silac_label2, traceback.format_exc()))

        data.to_csv('{}_stats'.format(out_path), sep=str('\t'), index=None)

    if html:
        html_map = []
        peak_map = {'header': [i[1] for i in PEAK_REPORTING], 'data': []}
        for index, (row_index, row) in enumerate(data.iterrows()):
            res = '\t'.join(row.astype(str))
            to_write, html_info = table_rows([{'table': res.strip(), 'html': html_data[index], 'keys': header_mapping}])
            html_map.append(html_info)
            peak_map['data'].append(peak_data[index])
            html_out.write(to_write)
            html_out.flush()

        template = []
        append = False
        for i in open(pyquant_html_file, 'r'):
            if 'HTML BREAK' in i:
                append = True
            elif append:
                template.append(i)
        html_template = Template(''.join(template))
        html_out.write(html_template.safe_substitute({
            'peak_output': base64.b64encode(gzip.zlib.compress(json.dumps(peak_map), 9)),
            'html_output': base64.b64encode(gzip.zlib.compress(json.dumps(html_map), 9)),
        }))

    os.remove(temp_file.name)
