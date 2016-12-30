from __future__ import division, unicode_literals, print_function
import sys
import os
import copy
import operator
import traceback

from functools import cmp_to_key

import pandas as pd
import numpy as np
import six

if six.PY3:
    xrange = range

from itertools import groupby, combinations
from collections import OrderedDict, defaultdict
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from multiprocessing import Process

try:
    from profilestats import profile
    from memory_profiler import profile as memory_profiler
except ImportError:
    pass

from scipy import integrate
from scipy.ndimage.filters import gaussian_filter1d

from pythomics.proteomics import config

from . import peaks
from .utils import calculate_theoretical_distribution, find_scan, find_prior_scan, find_next_scan, find_common_peak_mean, nanmean


class Worker(Process):
    def __init__(self, queue=None, results=None, precision=6, raw_name=None, mass_labels=None, isotope_ppms=None,
                 debug=False, html=False, mono=False, precursor_ppm=5.0, isotope_ppm=2.5, quant_method='integrate',
                 reader_in=None, reader_out=None, thread=None, fitting_run=False, msn_rt_map=None, reporter_mode=False,
                 spline=None, isotopologue_limit=-1, labels_needed=1, overlapping_mz=False, min_resolution=0, min_scans=3,
                 quant_msn_map=None, mrm=False, mrm_pair_info=None, peak_cutoff=0.05, ratio_cutoff=0, replicate=False,
                 ref_label=None, max_peaks=4, parser_args=None):
        super(Worker, self).__init__()
        self.precision = precision
        self.precursor_ppm = precursor_ppm
        self.isotope_ppm = isotope_ppm
        self.queue = queue
        self.reader_in, self.reader_out = reader_in, reader_out
        self.msn_rt_map = pd.Series(msn_rt_map)
        self.msn_rt_map.sort()
        self.results = results
        self.mass_labels = {'Light': {}} if mass_labels is None else mass_labels
        self.shifts = {0: "Light"}
        self.shifts.update(
            {sum(silac_masses.keys()): silac_label for silac_label, silac_masses in six.iteritems(self.mass_labels)})
        self.raw_name = raw_name
        self.filename = os.path.split(self.raw_name)[1]
        self.rt_tol = 0.2    # for fitting
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
        self.ratio_cutoff = ratio_cutoff
        self.ref_label = ref_label
        self.max_peaks = max_peaks
        self.parser_args = parser_args
        if mrm:
            self.quant_mrm_map = {label: list(group) for label, group in
                                                        groupby(self.quant_msn_map, key=operator.itemgetter(0))}
        self.peaks_n = self.parser_args.peaks_n
        self.rt_guide = not self.parser_args.no_rt_guide
        self.filter_peaks = not self.parser_args.disable_peak_filtering
        self.report_ratios = not self.parser_args.no_ratios
        self.bigauss_stepsize = 6 if self.parser_args.remove_baseline else 4

        # This is a convenience object to pass to the findAllPeaks function since it is called quite a few times

        self.peak_finding_kwargs = {
            'max_peaks': self.max_peaks,
            'debug': self.debug,
            'snr': self.parser_args.snr_filter,
            'amplitude_filter': self.parser_args.intensity_filter,
            'peak_width_end': self.parser_args.min_peak_separation,
            'baseline_correction': self.parser_args.remove_baseline,
            'zscore': self.parser_args.zscore_filter,
            'local_filter_size': self.parser_args.filter_width,
        }

    def get_calibrated_mass(self, mass):
        return mass / (1 - self.spline(mass) / 1e6) if self.spline else mass

    def low_snr(self, scan_intensities, thresh=0.3):
        std = np.std(scan_intensities)
        last_point = nanmean(scan_intensities[-3:])
        # check the SNR of the last points, if its bad, get out
        return (last_point / std) < thresh

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

        for i, v in common_peaks.items():
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
                    classifier = OneClassSVM(nu=0.95 * 0.15 + 0.05, kernel=str('linear'), degree=1, random_state=0)
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
                    x_mean, x_std1, x_std2 = np.median(data[classes == 1], axis=0)
            except IndexError:
                x_mean, x_std1, x_std2 = np.median(data, axis=0)
            else:
                x_inlier_indices = [i for i, v in enumerate(classes) if v in true_pred or common_peaks[keys[i][0]][keys[i][1]][keys[i][2]].get('valid')]
                x_inliers = set([keys[i][:2] for i in sorted(x_inlier_indices)])
                x_outliers = [i for i, v in enumerate(classes) if keys[i][:2] not in x_inliers and (
                    v in false_pred or common_peaks[keys[i][0]][keys[i][1]][keys[i][2]].get('interpolate'))]
                if debug:
                    print('inliers', x_inliers)
                    print('outliers', x_outliers)
                # print('x1o', x1_outliers)
                min_x = x_mean - x_std1
                max_x = x_mean + x_std2
                for index in x_inlier_indices:
                    indexer = keys[index]
                    peak_info = common_peaks[indexer[0]][indexer[1]][indexer[2]]
                    peak_min = peak_info['mean'] - peak_info['std']
                    peak_max = peak_info['mean'] + peak_info['std2']
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
            for quant_label, isotope_peaks in common_peaks.items():
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
        for i in sorted(set(to_delete), key=operator.itemgetter(0, 1, 2), reverse=True):
            del common_peaks[i[0]][i[1]][i[2]]
        return x_mean

    def convertScan(self, scan):
        import numpy as np
        scan_vals = scan['vals']
        res = pd.Series(scan_vals[:, 1].astype(np.uint64), index=np.round(scan_vals[:, 0], self.precision),
                                        name=int(scan['title']) if self.mrm else scan['rt'], dtype='uint64')
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
    # @line_profiler(extra_view=[peaks.findEnvelope, peaks.findAllPeaks, peaks.findMicro])
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

            combine_xics = scan_info.get('combine_xics')

            precursor = target_scan['precursor']
            calibrated_precursor = self.get_calibrated_mass(precursor)
            theor_mass = target_scan.get('theor_mass', calibrated_precursor)
            # this will be the RT of the target_scan, which is not always equal to the RT of the quant_scan
            rt = target_scan['rt']

            peptide = target_scan.get('peptide')
            if self.debug:
                sys.stderr.write('thread {4} on ms {0} {1} {2} {3}\n'.format(ms1, rt, precursor, scan_info, id(self)))

            result_dict = {
              'peptide': target_scan.get('mod_peptide', peptide),
              'scan': scanId,
              'ms1': ms1,
              'charge': charge,
              'modifications': target_scan.get('modifications'),
              'rt': rt,
              'accession': target_scan.get('accession')
            }
            if float(charge) == 0:
                # We cannot proceed with a zero charge
                self.results.put(result_dict)
                return

            precursors = defaultdict(dict)
            silac_dict = {'data': None, 'df': pd.DataFrame(), 'precursor': 'NA',
                                        'isotopes': {}, 'peaks': OrderedDict(), 'intensity': 'NA'}
            data = OrderedDict()
            # data['Light'] = copy.deepcopy(silac_dict)
            combined_data = pd.DataFrame()
            if self.mrm:
                mrm_labels = [i for i in self.mrm_pair_info.columns if i.lower() not in ('retention time')]
                mrm_info = None
                for index, values in self.mrm_pair_info.iterrows():
                    if values['Light'] == mass:
                        mrm_info = values
            for ion in target_scan.get('ion_set', []):
                precursors[str(ion)]['uncalibrated_mz'] = ion
                precursors[str(ion)]['calibrated_mz'] = self.get_calibrated_mass(ion)
                precursors[str(ion)]['theoretical_mz'] = ion
                data[str(ion)] = copy.deepcopy(silac_dict)
            for silac_label, silac_masses in self.mass_labels.items():
                silac_shift = 0
                global_mass = None
                added_residues = set([])
                cterm_mass = 0
                nterm_mass = 0
                mass_keys = list(silac_masses.keys())
                if self.reporter_mode:
                    silac_shift = sum(mass_keys)
                    label_mz = silac_shift
                    theo_mz = silac_shift
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
                    silac_shift += cterm_mass + nterm_mass

                    label_mz = precursor + (silac_shift / float(charge))
                    theo_mz = theor_mass + (silac_shift / float(charge))
                precursors[silac_label]['uncalibrated_mz'] = label_mz
                precursors[silac_label]['calibrated_mz'] = self.get_calibrated_mass(label_mz)
                precursors[silac_label]['theoretical_mz'] = theo_mz
                data[silac_label] = copy.deepcopy(silac_dict)
            if not precursors:
                precursors['Precursor']['uncalibrated_mz'] = precursor
                precursors['Precursor']['calibrated_mz'] = self.get_calibrated_mass(precursor)
                precursors['Precursor']['theoretical_mz'] = precursor
                data['Precursor'] = copy.deepcopy(silac_dict)
            precursors = OrderedDict(
                sorted(precursors.items(), key=cmp_to_key(lambda x, y: int(x[1]['uncalibrated_mz'] - y[1]['uncalibrated_mz']))))
            shift_maxes = {i: max([j['uncalibrated_mz'], j['calibrated_mz'], j['theoretical_mz']]) for i, j in
                                         zip(precursors.keys(), list(precursors.values())[1:])}
            lowest_precursor_mz = min(
                [label_val for label, label_info in precursors.items() for label_info_key, label_val in label_info.items() if
                 label_info_key.endswith('mz')])
            highest_precursor_mz = max(shift_maxes.values()) if shift_maxes else lowest_precursor_mz
            # do these here, remember when you tried to do this in one line with () and spent an hour debugging it?
            lowest_precursor_mz -= 5
            highest_precursor_mz += 5

            finished_isotopes = {i: set([]) for i in precursors.keys()}
            ms_index = 0
            delta = -1
            theo_dist = calculate_theoretical_distribution(peptide=peptide.upper()) if peptide else None
            spacing = config.NEUTRON / float(charge)
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
            all_data_intensity = {-1: [], 1: []}
            while True:
                map_to_search = self.quant_mrm_map[mass] if self.mrm else self.quant_msn_map
                if current_scan is None:
                    current_scan = initial_scan
                else:
                    if scans_to_quant:
                        current_scan = scans_to_quant.pop(0)
                    elif scans_to_quant is None:
                        current_scan = find_prior_scan(map_to_search, current_scan) if delta == -1 else find_next_scan(map_to_search, current_scan)
                    else:
                        # we've exhausted the scans we are supposed to quantify
                        break
                found = set([])
                current_scan_intensity = 0
                if current_scan is not None:
                    if current_scan in scans_to_skip:
                        continue
                    else:
                        df, scan_params = self.getScan(
                          current_scan,
                          start=None if self.mrm else lowest_precursor_mz,
                          end=None if self.mrm else highest_precursor_mz
                        )
                        # check if it's a low res scan, if so skip it
                        if self.min_resolution and df is not None:
                            scan_resolution = np.average(
                                df.index[1:] / np.array([df.index[i] - df.index[i - 1] for i in xrange(1, len(df))]))
                            if scan_resolution < self.min_resolution:
                                scans_to_skip.add(current_scan)
                                continue
                    if df is not None:
                        labels_found = set([])
                        xdata = df.index.values.astype(float)
                        ydata = df.fillna(0).values.astype(float)
                        iterator = precursors.items() if not self.mrm else [(mrm_label, 0)]
                        for precursor_label, precursor_info in iterator:
                            selected = {}
                            if self.mrm:
                                labels_found.add(precursor_label)
                                for i, j in zip(xdata, ydata):
                                    selected[i] = j
                                isotope_labels[df.name] = {
                                    'label': precursor_label,
                                    'isotope_index': target_scan.get('product_ion', 0),
                                }
                                key = (df.name, xdata[-1])
                                isotopes_chosen[key] = {
                                    'label': precursor_label,
                                    'isotope_index': target_scan.get('product_ion', 0),
                                    'amplitude': ydata[-1],
                                }
                            else:
                                uncalibrated_precursor = precursor_info['uncalibrated_mz']
                                measured_precursor = precursor_info['calibrated_mz']
                                theoretical_precursor = precursor_info['theoretical_mz']
                                data[precursor_label]['calibrated_precursor'] = measured_precursor
                                data[precursor_label]['precursor'] = uncalibrated_precursor
                                shift_max = shift_maxes.get(precursor_label) if self.overlapping_mz is False else None
                                is_fragmented_scan = (current_scan == initial_scan) and (precursor == measured_precursor)
                                envelope = peaks.findEnvelope(
                                    xdata,
                                    ydata,
                                    measured_mz=measured_precursor,
                                    theo_mz=theoretical_precursor,
                                    max_mz=shift_max,
                                    charge=charge,
                                    precursor_ppm=self.precursor_ppm,
                                    isotope_ppm=self.isotope_ppm,
                                    reporter_mode=self.reporter_mode,
                                    isotope_ppms=self.isotope_ppms if self.fitting_run else None,
                                    quant_method=self.quant_method,
                                    debug=self.debug,
                                    theo_dist=theo_dist if (self.mono or precursor_label not in shift_maxes) else None,
                                    label=precursor_label,
                                    skip_isotopes=finished_isotopes[precursor_label],
                                    last_precursor=last_precursors[delta].get(precursor_label, measured_precursor),
                                    isotopologue_limit=self.isotopologue_limit,
                                    fragment_scan=is_fragmented_scan,
                                    centroid=scan_params.get('centroid', False)
                                )
                                if not envelope['envelope']:
                                    if self.debug:
                                        print('envelope empty', envelope, measured_precursor, initial_scan, current_scan, last_precursors)
                                    if self.parser_args.msn_all_scans:
                                        selected[measured_precursor] = 0
                                        isotope_labels[measured_precursor] = {
                                          'label': precursor_label,
                                          'isotope_index': 0,
                                        }
                                        isotopes_chosen[(df.name, measured_precursor)] = {
                                          'label': precursor_label,
                                          'isotope_index': 0,
                                          'amplitude': 0,
                                        }
                                    else:
                                        continue

                                if not self.parser_args.msn_all_scans and 0 in envelope['micro_envelopes'] and \
                                                envelope['micro_envelopes'][0].get('int'):
                                    if ms_index == 0:
                                        last_precursors[delta * -1][precursor_label] = envelope['micro_envelopes'][0]['params'][1]
                                    last_precursors[delta][precursor_label] = envelope['micro_envelopes'][0]['params'][1]
                                added_keys = []
                                for isotope, vals in six.iteritems(envelope['micro_envelopes']):
                                    if isotope in finished_isotopes[precursor_label]:
                                        continue
                                    peak_intensity = vals.get('int')
                                    if peak_intensity == 0 or (self.peak_cutoff and peak_intensity < last_peak_height[precursor_label][isotope] * self.peak_cutoff):
                                        low_int_isotopes[(precursor_label, isotope)] += 1
                                        if not self.parser_args.msn_all_scans and low_int_isotopes[(precursor_label, isotope)] >= 2:
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
                                    selected[measured_precursor + isotope * spacing] = peak_intensity
                                    current_scan_intensity += peak_intensity
                                    vals['isotope'] = isotope
                                    isotope_labels[measured_precursor + isotope * spacing] = {
                                      'label': precursor_label,
                                      'isotope_index': isotope,
                                    }
                                    key = (df.name, measured_precursor + isotope * spacing)
                                    added_keys.append(key)
                                    isotopes_chosen[key] = {
                                        'label': precursor_label,
                                        'isotope_index': isotope,
                                        'amplitude': peak_intensity,
                                    }
                                del envelope
                            selected = pd.Series(selected, name=df.name).to_frame()
                            if df.name in combined_data.columns:
                                combined_data = combined_data.add(selected, axis='index', fill_value=0)
                            else:
                                combined_data = pd.concat([combined_data, selected], axis=1).fillna(0)
                            del selected
                        if not self.mrm and ((len(labels_found) < self.labels_needed) or (
                                            self.parser_args.require_all_ions and len(labels_found) < len(precursors))):
                            if self.parser_args.msn_all_scans:
                                if self.parser_args.require_all_ions:
                                    if self.debug:
                                        print('Not all ions found, setting', df.name, 'to zero')
                                    combined_data[df.name] = 0
                            else:
                                found.discard(precursor_label)
                                if df is not None and df.name in combined_data.columns:
                                    del combined_data[df.name]
                                    for i in isotopes_chosen.keys():
                                        if i[0] == df.name:
                                            del isotopes_chosen[i]
                        del df
                all_data_intensity[delta].append(current_scan_intensity)
                if not found or ((np.abs(ms_index) > 7 and self.low_snr(all_data_intensity[delta], thresh=self.parser_args.xic_snr)) or (self.parser_args.xic_window_size != -1 and np.abs(ms_index) >= self.parser_args.xic_window_size)):
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

            if self.parser_args.merge_labels or combine_xics:
                label_name = '_'.join(map(str, combined_data.index))
                combined_data = combined_data.sum(axis=0).to_frame(name=label_name).T
                isotope_labels = {
                    label_name: {
                        'isotope_index': 0,
                        'label': label_name,
                    }
                }
                data[label_name] = {}
                data[label_name]['calibrated_precursor'] = '_'.join(
                    map(str, (data[i].get('calibrated_precursor') for i in sorted(data.keys()) if i != label_name)))
                data[label_name]['precursor'] = '_'.join(
                    map(str, (data[i].get('precursor') for i in sorted(data.keys()) if i != label_name)))
            if isotopes_chosen and isotope_labels and not combined_data.empty:
                if self.mrm:
                    combined_data = combined_data.T
                # bookend with zeros if there aren't any, do the right end first because pandas will by default append there
                combined_data = combined_data.sort_index().sort_index(axis='columns')
                start_rt = rt
                rt_guide = self.rt_guide and start_rt
                if len(combined_data.columns) == 1:
                    if combined_data.columns[-1] == self.msn_rt_map[-1]:
                        new_col = combined_data.columns[-1] + (combined_data.columns[-1] - self.msn_rt_map[-2])
                    else:
                        try:
                            new_col = self.msn_rt_map.iloc[self.msn_rt_map.searchsorted(combined_data.columns[-1]) + 1].values[0]
                        except:
                            if self.debug:
                                print(combined_data.columns)
                                print(self.msn_rt_map)
                else:
                    new_col = combined_data.columns[-1] + (combined_data.columns[-1] - combined_data.columns[-2])
                combined_data[new_col] = 0
                new_col = combined_data.columns[0] - (combined_data.columns[1] - combined_data.columns[0])
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
                        'common-x': ['x'] + all_x,
                        'max-y': isotopes_chosen['amplitude'].max(),
                    }
                    isotope_figure_mapper = {}
                    rt_figure = {
                        'data': [],
                        'plot-multi': True,
                        'common-x': ['x'] + ['{0:0.4f}'.format(i) for i in combined_data.columns],
                        'rows': len(precursors),
                        'max-y': combined_data.max().max(),
                    }
                    rt_figure_mapper = {}

                    for counter, (index, row) in enumerate(isotope_group):
                        try:
                            title = 'Scan {} RT {}'.format(self.msn_rt_map[self.msn_rt_map == index].index[0], index)
                        except:
                            title = '{}'.format(index)
                        if index in isotope_figure_mapper:
                            isotope_base = isotope_figure_mapper[index]
                        else:
                            isotope_base = {'data': {'x': 'x', 'columns': [], 'type': 'bar'},
                                                            'axis': {'x': {'label': 'M/Z'}, 'y': {'label': 'Intensity'}}}
                            isotope_figure_mapper[index] = isotope_base
                            isotope_figure['data'].append(isotope_base)
                        for group in precursors.keys():
                            label_df = row[row['label'] == group]
                            x = label_df['amplitude'].index.get_level_values('MZ').tolist()
                            y = label_df['amplitude'].values.tolist()
                            isotope_base['data']['columns'].append(
                                ['{} {}'.format(title, group)] + [y[x.index(i)] if i in x else 0 for i in all_x])

                if not self.reporter_mode:
                    combined_peaks = defaultdict(dict)
                    peak_location = None

                    # If we are searching for a particular RT, we look for it in the data and remove other larger peaks
                    # until we find it. To help with cases where we are fitting multiple datasets for the same XIC, we
                    # combine the data to increase the SNR in case some XICs of a given ion are weak

                    if rt_guide and not self.parser_args.msn_all_scans:
                        merged_data = combined_data.sum(axis=0)
                        merged_x = merged_data.index.astype(float).values
                        merged_y = merged_data.values.astype(float)
                        res, residual = peaks.targeted_search(
                            merged_x,
                            merged_y,
                            start_rt,
                            attempts=4,
                            stepsize=self.bigauss_stepsize,
                            peak_finding_kwargs=self.peak_finding_kwargs
                        )

                        if self.debug:
                            if res is not None:
                                print('peak used for sub-fitting', res)
                            else:
                                print(peptide, 'is dead')

                        if res is not None:
                            rt_means = res[1::self.bigauss_stepsize]
                            rt_amps = res[::self.bigauss_stepsize]
                            rt_std = res[2::self.bigauss_stepsize]
                            rt_std2 = res[3::self.bigauss_stepsize]
                            m_std = np.std(merged_y)
                            m_mean = nanmean(merged_y)
                            valid_peaks = [
                                {'mean': i, 'amp': j, 'std': l, 'std2': k, 'total': merged_y.sum(), 'snr': m_mean / m_std,
                                 'residual': residual}
                                for i, j, l, k in zip(rt_means, rt_amps, rt_std, rt_std2)]

                            valid_peaks.sort(key=lambda x: np.abs(x['mean'] - start_rt))

                            peak_index = peaks.find_nearest_index(merged_x, valid_peaks[0]['mean'])
                            peak_location = merged_x[peak_index]
                            if self.debug:
                                print('peak location is', peak_location)
                            merged_lb = peaks.find_nearest_index(merged_x, valid_peaks[0]['mean'] - valid_peaks[0]['std'] * 2)
                            merged_rb = peaks.find_nearest_index(merged_x, valid_peaks[0]['mean'] + valid_peaks[0]['std2'] * 2)
                            merged_rb = len(merged_x) if merged_rb == -1 else merged_rb + 1
                        else:
                            merged_lb = 0
                            merged_rb = combined_data.shape[1]

                    else:
                        merged_x = xdata
                        merged_y = ydata
                        merged_lb = 0
                        merged_rb = combined_data.shape[1]

                    for row_num, (index, values) in enumerate(combined_data.iterrows()):
                        quant_label = isotope_labels.loc[index, 'label']
                        xdata = values.index.values.astype(float)
                        ydata = values.fillna(0).values.astype(float)
                        if sum(ydata > 0) >= self.min_scans:
                            # this step is to add in a term on the border if possible
                            # otherwise, there are no penalties on the variance if it is
                            # at the border since the data does not exist. We only add for lower values to avoid
                            # including monster peaks we may be explicitly excluding above
                            fit_lb = merged_lb
                            fit_rb = merged_rb
                            while fit_rb + 1 < len(ydata) and ydata[fit_rb + 1] <= ydata[fit_rb - 1]:
                                fit_rb += 1
                            while fit_lb != 0 and ydata[fit_lb] >= ydata[fit_lb - 1]:
                                fit_lb -= 1
                            peak_x = np.copy(xdata[fit_lb:fit_rb])
                            peak_y = np.copy(ydata[fit_lb:fit_rb])
                            if peak_x.size <= 1 or sum(peak_y > 0) < self.min_scans:
                                continue
                            if rt_guide:
                                peak_positive_y = peak_y > 0
                                if peak_location is None and self.parser_args.msn_all_scans:
                                    peak_location = start_rt
                                nearest_positive_peak = peaks.find_nearest(peak_x[peak_positive_y], peak_location)
                                sub_peak_location = peaks.find_nearest_index(peak_x, nearest_positive_peak)
                                sub_peak_index = sub_peak_location if peak_y[sub_peak_location] else np.argmax(peak_y)
                            else:
                                nearest_positive_peak = 0
                            # fit, residual = peaks.fixedMeanFit2(peak_x, peak_y, peak_index=sub_peak_index, debug=self.debug)
                            if self.debug:
                                print('fitting XIC for', quant_label, index)
                                print('raw data is', xdata.tolist(), ydata.tolist())
                            fit, residual = peaks.findAllPeaks(
                              xdata,
                              ydata,
                              bigauss_fit=True,
                              filter=self.filter_peaks,
                              rt_peak=nearest_positive_peak,
                              peak_width_start=1,
                              **self.peak_finding_kwargs
                            )
                            if not fit.any():
                                continue
                            rt_amps = fit[::self.bigauss_stepsize]    # * ydata.max()
                            rt_means = fit[1::self.bigauss_stepsize]
                            rt_std = fit[2::self.bigauss_stepsize]
                            rt_std2 = fit[3::self.bigauss_stepsize]
                            xic_peaks = []
                            positive_y = ydata[ydata > 0]
                            if len(positive_y) > 5:
                                positive_y = gaussian_filter1d(positive_y, 3, mode='constant')
                            for i, j, l, k in zip(rt_means, rt_amps, rt_std, rt_std2):
                                d = {
                                    'mean': i,
                                    'amp': j,
                                    'std': l,
                                    'std2': k,
                                    'total': values.sum(),
                                    'residual': residual,
                                }
                                mean_index = peaks.find_nearest_index(xdata[ydata > 0], i)
                                window_size = 5 if len(positive_y) < 15 else int(len(positive_y) / 3)
                                lb, rb = mean_index - window_size, mean_index + window_size + 1
                                if lb < 0:
                                    lb = 0
                                if rb > len(positive_y):
                                    rb = -1
                                data_window = positive_y[lb:rb]
                                if data_window.any():
                                    try:
                                        background = np.percentile(data_window, 0.8)
                                    except:
                                        background = np.percentile(ydata, 0.8)
                                    mean = nanmean(data_window)
                                    if background < mean:
                                        background = mean
                                    d['sbr'] = nanmean(j / (np.array( sorted(data_window, reverse=True)[:5])))    # (j-np.mean(positive_y[lb:rb]))/np.std(positive_y[lb:rb])
                                    if len(data_window) > 2:
                                        d['snr'] = (j - background) / np.std(data_window)
                                    else:
                                        d['snr'] = np.NaN
                                else:
                                    d['sbr'] = np.NaN
                                    d['snr'] = np.NaN
                                xic_peaks.append(d)
                            # if we have a peaks containing our retention time, keep them and throw out ones not containing it
                            to_remove = []
                            to_keep = []
                            if rt_guide:
                                peak_location_index = peaks.find_nearest_index(merged_x, peak_location)
                                for i, v in enumerate(xic_peaks):
                                    mu = v['mean']
                                    s1 = v['std']
                                    s2 = v['std2']
                                    if mu - s1 * 2 < start_rt < mu + s2 * 2:
                                        # these peaks are considered true and will help with the machine learning
                                        if mu - s1 * 1.5 < start_rt < mu + s2 * 1.5:
                                            v['valid'] = True
                                            to_keep.append(i)
                                    elif np.abs(peaks.find_nearest_index(merged_x, mu) - peak_location_index) > 2:
                                        to_remove.append(i)
                            # kick out peaks not containing our RT
                            if rt_guide:
                                if not to_keep:
                                    # we have no peaks with our RT, there are contaminating peaks, remove all the noise but the closest to our RT
                                    if not self.mrm:
                                        # for i in to_remove:
                                        #         xic_peaks[i]['interpolate'] = True
                                        valid_peak = \
                                            sorted([(i, np.abs(i['mean'] - start_rt)) for i in xic_peaks], key=operator.itemgetter(1))[0][0]
                                        for i in reversed(xrange(len(xic_peaks))):
                                            if xic_peaks[i] == valid_peak:
                                                continue
                                            else:
                                                del xic_peaks[i]
                                                # valid_peak['interpolate'] = True
                                                # else:
                                                #         valid_peak = [j[0] for j in sorted([(i, i['amp']) for i in xic_peaks], key=operator.itemgetter(1), reverse=True)[:3]]
                                else:
                                    # if not to_remove:
                                    #         xic_peaks = [xic_peaks[i] for i in to_keep]
                                    # else:
                                    for i in reversed(to_remove):
                                        del xic_peaks[i]
                            if self.debug:
                                print(quant_label, index)
                                print(fit)
                                print(to_remove, to_keep, xic_peaks)
                            combined_peaks[quant_label][index] = xic_peaks    # if valid_peak is None else [valid_peak]

                        if self.html:
                            # ax = fig.add_subplot(subplot_rows, subplot_columns, fig_index)
                            if quant_label in rt_figure_mapper:
                                rt_base = rt_figure_mapper[(quant_label, index)]
                            else:
                                rt_base = {
                                    'data': {
                                        'x': 'x',
                                        'columns': []
                                    },
                                    'grid': {
                                        'x': {
                                            'lines': [{
                                                'value': rt,
                                                'text': 'Initial RT {0:0.4f}'.format(rt),
                                                'position': 'middle'
                                            }]
                                        }
                                    },
                                    'subchart': {
                                        'show': True
                                    },
                                    'axis': {
                                        'x': {
                                            'label': 'Retention Time'
                                        },
                                        'y': {
                                            'label': 'Intensity'
                                        }
                                    }
                                }
                                rt_figure_mapper[(quant_label, index)] = rt_base
                                rt_figure['data'].append(rt_base)
                            rt_base['data']['columns'].append(['{0} {1} raw'.format(quant_label, index)] + ydata.tolist())

                peak_info = {i: {} for i in self.mrm_pair_info.columns} if self.mrm else {i: {} for i in precursors.keys()}
                if self.reporter_mode or combined_peaks:
                    if self.reporter_mode:
                        for row_num, (index, values) in enumerate(combined_data.iterrows()):
                            quant_label = isotope_labels.loc[index, 'label']
                            isotope_index = isotope_labels.loc[index, 'isotope_index']
                            int_val = sum(values)
                            quant_vals[quant_label][isotope_index] = int_val
                    else:
                        # common_peak = self.replaceOutliers(combined_peaks, combined_data, debug=self.debug)
                        common_peak = find_common_peak_mean(combined_peaks)
                        common_loc = peaks.find_nearest_index(xdata, common_peak)    # np.where(xdata==common_peak)[0][0]
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
                                closest_rts = sorted([(i, np.abs(i['mean'] - common_peak)) for i in values], key=operator.itemgetter(1))
                                xic_peaks = [i[0] for i in closest_rts]
                                pos_x = xdata[ydata > 0]
                                if rt_guide:
                                    xic_peaks = [xic_peaks[0]]
                                else:
                                    # unguided, sort by amplitude
                                    xic_peaks.sort(key=operator.itemgetter('amp'), reverse=True)
                                for xic_peak_index, xic_peak in enumerate(xic_peaks):
                                    if self.peaks_n != -1 and xic_peak_index >= self.peaks_n:    # xic_peak index is 0 based, peaks_n is 1 based, hence the >=
                                        break
                                    # if we move more than a # of ms1 to the dominant peak, update to our known peak
                                    gc = 'k'
                                    nearest = peaks.find_nearest_index(pos_x, xic_peak['mean'])
                                    peak_loc = np.where(xdata == pos_x[nearest])[0][0]
                                    mean = xic_peak['mean']
                                    amp = xic_peak['amp']
                                    mean_diff = mean - xdata[common_loc]
                                    mean_diff = np.abs(mean_diff / xic_peak['std'] if mean_diff < 0 else mean_diff / xic_peak['std2'])
                                    std = xic_peak['std']
                                    std2 = xic_peak['std2']
                                    snr = xic_peak['snr']
                                    sbr = xic_peak['sbr']
                                    residual = xic_peak['residual']
                                    if False and len(xdata) >= 3 and (mean_diff > 2 or (np.abs(peak_loc - common_loc) > 2 and mean_diff > 2)):
                                        # fixed mean fit
                                        if self.debug:
                                            print(quant_label, index)
                                            print(common_loc, peak_loc)
                                        nearest = peaks.find_nearest_index(pos_x, mean)
                                        nearest_index = np.where(xdata == pos_x[nearest])[0][0]
                                        res = peaks.fixedMeanFit(xdata, ydata, peak_index=nearest_index, debug=self.debug)
                                        if res is None:
                                            if self.debug:
                                                print(quant_label, index, 'has no values here')
                                            continue
                                        amp, mean, std, std2 = res
                                        amp *= ydata.max()
                                        gc = 'g'
                                    # var_rat = closest_rt['var']/common_var
                                    peak_params = np.array([amp, mean, std, std2])
                                    # int_args = (res.x[rt_index]*mval, res.x[rt_index+1], res.x[rt_index+2])
                                    left, right = xdata[0] - 4 * std, xdata[-1] + 4 * std2
                                    xr = np.linspace(left, right, 1000)
                                    left_index, right_index = peaks.find_nearest_index(xdata, left), peaks.find_nearest_index(xdata, right) + 1
                                    if left_index < 0:
                                        left_index = 0
                                    if right_index >= len(xdata) or right_index <= 0:
                                        right_index = len(xdata)

                                    # check that we have at least 2 positive values
                                    if sum(ydata[left_index:right_index] > 0) < 2:
                                        continue

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
                                    sdr = np.log2(int_val * 1. / total_int + 1.)

                                    if int_val and not pd.isnull(int_val) and gc != 'c':
                                        try:
                                            quant_vals[quant_label][isotope_index] += int_val
                                        except KeyError:
                                            try:
                                                quant_vals[quant_label][isotope_index] = int_val
                                            except KeyError:
                                                quant_vals[quant_label] = {isotope_index: int_val}
                                    cleft, cright = mean - 2 * std, mean + 2 * std2
                                    curve_indices = (xdata >= cleft) & (xdata <= cright)
                                    cf_data = ydata[curve_indices]
                                    # Buffer cf_data with 0's to reflect that the data is nearly zero outside the fit
                                    # and to prevent areas with 2 data points from having negative R^2
                                    cf_data = np.hstack((0, cf_data, 0))
                                    ss_tot = np.sum((cf_data - nanmean(cf_data)) ** 2)
                                    ss_res = np.sum((cf_data - np.hstack((0, peaks.bigauss_ndim(xdata[curve_indices], peak_params), 0))) ** 2)
                                    coef_det = 1 - ss_res / ss_tot
                                    peak_info_dict = {
                                        'mean': mean,
                                        'std': std,
                                        'std2': std2,
                                        'amp': amp,
                                        'mean_diff': mean_diff,
                                        'snr': snr,
                                        'sbr': sbr,
                                        'sdr': sdr,
                                        'auc': int_val,
                                        'peak_width': std + std2,
                                        'coef_det': coef_det,
                                        'residual': residual,
                                        'label': quant_label,
                                    }
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
                                        for i, v in enumerate(rt_base['data']['columns']):
                                            if key in v[0]:
                                                break
                                        rt_base['data']['columns'].insert(i, ['{0} {1} fit {2}'.format(quant_label, index, xic_peak_index)] + np.nan_to_num(peaks.bigauss_ndim(xdata, peak_params)).tolist())
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
                        for num_of_isotopes in xrange(2, len(isotopes) + 1):
                            for combo in combinations(isotopes, num_of_isotopes):
                                chosen_isotopes = np.array([isotope_ints[i] for i in combo])
                                chosen_isotopes /= chosen_isotopes.max()
                                chosen_dist = np.array([theo_dist[i] for i in combo])
                                chosen_dist /= chosen_dist.max()
                                res = sum((chosen_dist - chosen_isotopes) ** 2)
                                isotope_residuals.append((res, combo))
                        # this weird sorting is to put the favorable values as the lowest values
                        if isotope_residuals:
                            kept_keys = \
                                sorted(isotope_residuals, key=lambda x: (0 if x[0] < 0.1 else 1, len(isotopes) - len(x[1]), x[0]))[0][1]
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
                                    l1, l2 = 0, 0
                                    for i in common_isotopes:
                                        q1 = qv1.get(i)
                                        q2 = qv2.get(i)
                                        if q1 > 100 and q2 > 100 and q1 > l1 * 0.15 and q2 > l2 * 0.15:
                                            x.append(i)
                                            y.append(q1 / q2)
                                            l1, l2 = q1, q2
                                    # fit it and take the intercept
                                    if len(x) >= 3 and np.std(np.log2(y)) > 0.3:
                                        classifier = EllipticEnvelope(contamination=0.25, random_state=0)
                                        fit_data = np.log2(np.array(y).reshape(len(y), 1))
                                        true_pred = (True, 1)
                                        classifier.fit(fit_data)
                                        ratio = nanmean([y[i] for i, v in enumerate(classifier.predict(fit_data)) if v in true_pred])
                                    else:
                                        ratio = nanmean(np.array(y))
                                else:
                                    common_isotopes = set(qv1.keys()).union(qv2.keys())
                                    quant1 = sum([qv1.get(i, 0) for i in common_isotopes])
                                    quant2 = sum([qv2.get(i, 0) for i in common_isotopes])
                                    ratio = quant1 / quant2 if quant1 and quant2 else 'NA'
                                try:
                                    if self.ratio_cutoff and not pd.isnull(ratio) and np.abs(np.log2(ratio)) > self.ratio_cutoff:
                                        write_html = True
                                except:
                                    pass
                            result_dict.update({'{}_{}_ratio'.format(silac_label1, silac_label2): ratio})

                if write_html:
                    result_dict.update({'html_info': html_images})
                for peak_label, peak_data in six.iteritems(peak_info):
                    result_dict.update({
                        '{}_peaks'.format(peak_label): peak_data,
                        '{}_isotopes'.format(peak_label): sum((isotopes_chosen['label'] == peak_label) & (isotopes_chosen['amplitude'] > 0)),
                    })
            for silac_label, silac_data in six.iteritems(data):
                precursor = silac_data['precursor']
                calc_precursor = silac_data.get('calibrated_precursor', silac_data['precursor'])
                result_dict.update({
                    '{}_residual'.format(silac_label): nanmean(pd.Series(silac_data.get('residual', [])).replace([np.inf, -np.inf, np.nan], 0)),
                    '{}_precursor'.format(silac_label): precursor,
                    '{}_calibrated_precursor'.format(silac_label): calc_precursor,
                })
            result_dict.update({
                'ions_found': target_scan.get('ions_found'),
                'html': {
                    'xic': rt_figure,
                    'isotope': isotope_figure,
                }
            })
            self.results.put(result_dict)
            del result_dict
            del combined_data
            del isotopes_chosen
        except:
            print('ERROR encountered. Please report at https://github.com/Chris7/pyquant/issues:\n {}'.format(traceback.format_exc()))
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
