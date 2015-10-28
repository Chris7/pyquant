#!/usr/bin/env python
from __future__ import division, unicode_literals
import pyximport; pyximport.install()
# TODO: Threading for single files. Since much time is spent in fetching m/z records, it may be pointless
# because I/O is limiting. If so, create a thread just for I/O for processing requests the other threads
# interface with

description = """
This will quantify labeled peaks (such as SILAC) in ms1 spectra. It relies solely on the distance between peaks,
 which can correct for errors due to amino acid conversions.
"""
import sys
from string import Template
import gzip
import base64
import json
import decimal
import csv
import math
import os
import copy
import operator
import traceback
import pandas as pd
import numpy as np
import re
import random
from itertools import groupby
from collections import OrderedDict, defaultdict
from matplotlib import pyplot as plt
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.linear_model import RANSACRegressor
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Process, Queue, Manager, Array
import ctypes

try:
    from profilestats import profile
    from memory_profiler import profile as memory_profiler
except ImportError:
    pass
from Queue import Empty

import argparse
from datetime import datetime, timedelta
from scipy import integrate
from scipy.stats import linregress
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline

from pythomics.templates import CustomParser
from pythomics.proteomics.parsers import GuessIterator
from pythomics.proteomics import config
import peaks

RESULT_ORDER = [('peptide', 'Peptide'), ('modifications', 'Modifications'),
                ('charge', 'Charge'), ('ms1', 'MS1 Spectrum ID'), ('scan', 'MS2 Spectrum ID'), ('rt', 'Retention Time')]

ION_CUTOFF = 2


parser = CustomParser(description=description)
raw_group = parser.add_argument_group("Raw Data Parameters")
raw_group.add_argument('--scan-file', help="The scan file(s) for the raw data. If not provided, assumed to be in the directory of the processed/tabbed/peaklist file.", type=argparse.FileType('r'), nargs='*')
raw_group.add_argument('--scan-file-dir', help="The directory containing raw data.", type=str)
raw_group.add_argument('--precision', help="The precision for storing m/z values. Defaults to 6 decimal places.", type=int, default=6)
raw_group.add_argument('--precursor-ppm', help="The mass accuracy for the first monoisotopic peak in ppm.", type=float, default=5)
raw_group.add_argument('--isotope-ppm', help="The mass accuracy for the isotopic cluster.", type=float, default=2.5)
raw_group.add_argument('--spread', help="Assume there is spread of the isotopic label.", action='store_true')

search_group = parser.add_argument_group("Search Information")
parser.add_processed_ms(group=search_group, required=False)
search_group.add_argument('--skip', help="If true, skip scans with missing files in the mapping.", action='store_true')
search_group.add_argument('--peptide', help="The peptide(s) to limit quantification to.", type=str, nargs='*')
search_group.add_argument('--scan', help="The scan(s) to limit quantification to.", type=str, nargs='*')

replicate_group = parser.add_argument_group("Replicate Search")
replicate_group.add_argument('--replicate', help="Analyze files in replicate mode. This means the dataset being quantified is from the search file of a replicate", action='store_true')
replicate_group.add_argument('--rt-window', help="The maximal deviation of a scan's retention time to be considered for analysis.", default=0.25, type=float)

label_group = parser.add_argument_group("Labeling Information")
label_subgroup = label_group.add_mutually_exclusive_group()
label_subgroup.add_argument('--label-scheme', help='The file corresponding to the labeling scheme utilized.', type=argparse.FileType('r'))
label_subgroup.add_argument('--label-method', help='Predefined labeling schemes to use.', type=str, choices=sorted(config.MS1_SCHEMES.keys()))

tsv_group = parser.add_argument_group('Tabbed File Input')
tsv_group.add_argument('--tsv', help='A delimited file containing scan information.', type=argparse.FileType('r'))
tsv_group.add_argument('--label', help='The column indicating the label state of the peptide. If not found, entry assumed to be light variant.', default='Labeling State')
tsv_group.add_argument('--peptide-col', help='The column indicating the peptide.', default='Sequence')
tsv_group.add_argument('--rt', help='The column indicating the retention time.', default='Retention time')
tsv_group.add_argument('--mz', help='The column indicating the MZ value of the precursor ion. This is not the MH+.', default='m/z')
tsv_group.add_argument('--scan-col', help='The column indicating the scan corresponding to the ion.', default='MS/MS Scan Number')
tsv_group.add_argument('--charge', help='The column indicating the charge state of the ion.', default='Charge')
tsv_group.add_argument('--source', help='The column indicating the raw file the scan is contained in.', default='Raw file')

ion_search_group = parser.add_argument_group('Targetted Ion Search Parameters')
ion_search_group.add_argument('--msn', help='The ms level to search for the ion in. Default: 2 (ms2)', type=int, default=2)
ion_search_group.add_argument('--msn-ion', help='M/Z values to search for in the scans.', nargs='+', type=float)
ion_search_group.add_argument('--msn-peaklist', help='A file containing peaks to search for in the scans.', type=argparse.FileType('rb'))
ion_search_group.add_argument('--msn-ppm', help='The error tolerance for identifying the ion(s).', type=float, default=200)
ion_search_group.add_argument('--msn-quant-from', help='The ms level to quantify values from. i.e. if we are identifying an ion in ms2, we can quantify it in ms1 (or ms2). Default: msn value-1', type=int, default=None)

quant_parameters = parser.add_argument_group('Quantification Parameters')
quant_parameters.add_argument('--quant-method', help='The process to use for quantification. Default: Integrate for ms1, sum for ms2+.', choices=['integrate', 'sum'], default=None)
quant_parameters.add_argument('--reporter-ion', help='Indicates that reporter ions are being used. As such, we only analyze a single scan.', action='store_true')
quant_parameters.add_argument('--isotopologue-limit', help='How many isotopologues to quantify', type=int, default=-1)
quant_parameters.add_argument('--overlapping-mz', help='This declares the mz values will overlap. It is useful for data such as neucode, but not needed for only SILAC labeling.', action='store_true')
quant_parameters.add_argument('--labels-needed', help='How many labels need to be detected to quantify a scan (ie if you have a 2 state experiment and set this to 2, it will only quantify scans where both occur.', default=1, type=int)
quant_parameters.add_argument('--min-scans', help='How many quantification scans are needed to quantify a scan.', default=1, type=int)
quant_parameters.add_argument('--min-resolution', help='The minimal resolving power of a scan to consider for quantification. Useful for skipping low-res scans', default=0, type=float)
quant_parameters.add_argument('--no-mass-accuracy-correction', help='Disables the mass accuracy correction.', action='store_true')
quant_parameters.add_argument('--peak-cutoff', help='The threshold from the initial retention time a peak can fall by before being discarded', type=float, default=0.05)

mrm_parameters = parser.add_argument_group('SRM/MRM Parameters')
mrm_parameters.add_argument('--mrm-map', help='A file indicating light and heavy peptide pairs, and optionally the known elution time.', type=argparse.FileType('r'))

output_group = parser.add_argument_group("Output Options")
output_group.add_argument('--debug', help="This will output debug information and graphs.", action='store_true')
output_group.add_argument('--html', help="Output a HTML table summary.", action='store_true')
output_group.add_argument('--resume', help="Will resume from the last run. Only works if not directing output to stdout.", action='store_true')
output_group.add_argument('--sample', help="How much of the data to sample. Enter as a decimal (ie 1.0 for everything, 0.1 for 10%%)", type=float, default=1.0)
output_group.add_argument('--disable-stats', help="Disable confidence statistics on data.", action='store_true')
output_group.add_argument('-o', '--out', nargs='?', help='The prefix for the file output', type=str)

convenience_group = parser.add_argument_group('Convenience Parameters')
convenience_group.add_argument('--neucode', help='This will select parameters specific for neucode. Note: You still must define a labeling scheme.')
convenience_group.add_argument('--isobaric-tags', help='This will select parameters specific for isobaric tag based labeling (TMT/iTRAQ).')
convenience_group.add_argument('--mrm', help='This will select parameters specific for Selective/Multiple Reaction Monitoring (SRM/MRM).', action='store_true')


class Reader(Process):
    def __init__(self, incoming, outgoing, raw_file=None, spline=None):
        super(Reader, self).__init__()
        self.incoming = incoming
        self.outgoing = outgoing
        self.scan_dict = {}
        self.access_times = {}
        self.raw = raw_file
        self.spline = spline

    def run(self):
        for scan_request in iter(self.incoming.get, None):
            thread, scan_id, mz_start, mz_end = scan_request
            d = self.scan_dict.get(scan_id)
            if not d:
                scan = self.raw.getScan(scan_id)
                scan_vals = np.array(scan.scans)
                if self.spline:
                    scan_vals[:,0] = scan_vals[:,0]/(1-self.spline(scan_vals[:,0])/1e6)
                # add to our database
                d = {'vals': scan_vals, 'rt': scan.rt, 'title': scan.title, 'mass': scan.mass, 'charge': scan.charge}
                self.scan_dict[scan_id] = d
                # the scan has been stored, delete it
                del scan
            if mz_start is not None or mz_end is not None:
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
    def __init__(self, queue=None, results=None, precision=6, raw_name=None, silac_labels=None, isotope_ppms=None,
                 debug=False, html=False, mono=False, precursor_ppm=5.0, isotope_ppm=2.5, quant_method='integrate',
                 reader_in=None, reader_out=None, thread=None, fitting_run=False, msn_rt_map=None, reporter_mode=False,
                 spline=None, isotopologue_limit=-1, labels_needed=1, overlapping_mz=False, min_resolution=0, min_scans=3,
                 quant_msn_map=None, mrm=False, mrm_pair_info=None, peak_cutoff=0.05, ratio_cutoff=1, replicate=False):
        super(Worker, self).__init__()
        self.precision = precision
        self.precursor_ppm = precursor_ppm
        self.isotope_ppm = isotope_ppm
        self.queue=queue
        self.reader_in, self.reader_out = reader_in, reader_out
        self.msn_rt_map = pd.Series(msn_rt_map)
        self.msn_rt_map.sort()
        self.results = results
        self.silac_labels = {'Light': {}} if silac_labels is None else silac_labels
        self.shifts = {0: "Light"}
        self.shifts.update({sum(silac_masses.keys()): silac_label for silac_label, silac_masses in self.silac_labels.iteritems()})
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
        if mrm:
            self.quant_mrm_map = {label: list(group) for label, group in groupby(self.quant_msn_map, key=operator.itemgetter(0))}

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

    def replaceOutliers(self, common_peaks, combined_data):
        x = []
        y = []
        hx = []
        hy = []
        keys = []
        hkeys = []
        y2 = []
        hy2 = []
        for i,v in common_peaks.items():
            for isotope, peaks in v.items():
                for peak_index, peak in enumerate(peaks):
                    keys.append((i,isotope,peak_index))
                    x.append(peak['mean'])
                    y.append(peak['std'])
                    y2.append(peak['std2'])
                    if self.mrm and i != 'Light':
                        hx.append(peak['mean'])
                        hy.append(peak['std'])
                        hy2.append(peak['std2'])

                        hkeys.append((i, isotope, peak_index))
        classifier = EllipticEnvelope(support_fraction=0.75, random_state=0)
        if len(x) == 1:
            return x[0]
        data = np.array([x,y]).T
        false_pred = (False, -1)
        true_pred = (True, 1)
        to_delete = set([])
        try:
            classifier.fit(np.array([hx,hy]).T if self.mrm else data)
            x1_mean, x1_std = classifier.location_
        except ValueError:
            x1_mean, x1_std = data[0,0], data[0,1]
        else:
            classes = classifier.predict(data)
            x1_inliers = set([keys[i][:2] for i,v in enumerate(classes) if v in true_pred or common_peaks[keys[i][0]][keys[i][1]][keys[i][2]].get('valid')])
            # print x1_inliers
            x1_outliers = [i for i,v in enumerate(classes) if v in false_pred or (common_peaks[keys[i][0]][keys[i][1]][keys[i][2]].get('interpolate') and keys[i][:2] not in x1_inliers)]
            if x1_inliers:
                for index in x1_outliers:
                    indexer = keys[index]
                    if indexer[:2] in x1_inliers:
                        to_delete.add(indexer)
                    # elif common_peaks[indexer[i][0]][indexer[i][1]][indexer[i][2]].get('interpolate'):
                    #     mz = indexer[1]
                    #     row_data = combined_data.loc[mz, :]
                    #     mapper = interp1d(row_data.index.values, row_data.values)
                    #     common_peaks[indexer[0]][indexer[1]][indexer[2]]['amp'] = mapper(x1_mean)
                    #     common_peaks[indexer[0]][indexer[1]][indexer[2]]['peak'] = x1_mean
                    #     common_peaks[indexer[0]][indexer[1]][indexer[2]]['mean'] = x1_mean
                    #     common_peaks[indexer[0]][indexer[1]][indexer[2]]['std'] = x1_std
                    # else:
                    #     to_delete.add(indexer)
        data = np.array([x, y2]).T
        try:
            classifier.fit(np.array([hx,hy2]).T if self.mrm else data)
        except ValueError:
            pass
        else:
            classes = classifier.predict(data)
            x2_inliers = set([keys[i][:2] for i,v in enumerate(classes) if v in true_pred])
            x2_outliers = [i for i,v in enumerate(classes) if v in false_pred or (common_peaks[keys[i][0]][keys[i][1]][keys[i][2]].get('interpolate') and keys[i][:2] not in x2_inliers)]
            if x2_inliers:
                x2_mean, x2_std = classifier.location_
                for index in x2_outliers:
                    indexer = keys[index]
                    if indexer[:2] in x2_inliers:
                        if indexer[:2] not in x1_inliers:
                            to_delete.add(indexer)
                        # elif common_peaks[indexer[i][0]][indexer[i][1]][indexer[i][2]].get('interpolate'):
                        #     common_peaks[indexer[0]][indexer[1]][indexer[2]]['std2'] = x2_std
                        # else:
                        #     to_delete.add(indexer)
        # print to_delete
        for i in sorted(set(to_delete), key=operator.itemgetter(0,1,2), reverse=True):
            del common_peaks[i[0]][i[1]][i[2]]
        return x1_mean

    def convertScan(self, scan):
        import numpy as np
        scan_vals = scan['vals']
        res = pd.Series(scan_vals[:, 1].astype(np.uint64), index=np.round(scan_vals[:, 0], self.precision), name=int(scan['title']) if self.mrm else scan['rt'], dtype='uint64')
        del scan_vals
        # due to precision, we have multiple m/z values at the same place. We can eliminate this by grouping them and summing them.
        # Summation is the correct choice here because we are combining values of a precision higher than we care about.
        try:
            return res.groupby(level=0).sum() if not res.empty else None
        except:
            sys.stderr.write('Converting scan error {}\n{}\n{}\n'.format(traceback.format_exc(), res, scan))

    def getScan(self, ms1, start=None, end=None):
        self.reader_in.put((self.thread, ms1, start, end))
        scan = self.reader_out.get()
        return self.convertScan(scan)

    # @memory_profiler
    def quantify_peaks(self, params):
        try:
            html_images = {}
            scan_info = params.get('scan_info')
            target_scan = scan_info.get('id_scan')
            quant_scan = scan_info.get('quant_scan')
            scanId = target_scan.get('id')
            ms1 = quant_scan['id']
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
            data['Light'] = copy.deepcopy(silac_dict)
            combined_data = pd.DataFrame()
            highest_shift = 20
            if self.mrm:
                mrm_labels = [i for i in self.mrm_pair_info.columns if i.lower() not in ('retention time')]
                mrm_info = None
                for index, values in self.mrm_pair_info.iterrows():
                    if values['Light'] == mass:
                        mrm_info = values
            for silac_label, silac_masses in self.silac_labels.items():
                silac_shift=0
                global_mass = None
                added_residues = set([])
                cterm_mass = 0
                nterm_mass = 0
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
                    silac_shift += silac_masses.keys()[0]
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
            shift_maxes = {i: j for i,j in zip(precursors.keys(), precursors.values()[1:])}
            finished = set([])
            finished_isotopes = {i: set([]) for i in precursors.keys()}
            result_dict = {'peptide': target_scan.get('mod_peptide', peptide),
                           'scan': scanId, 'ms1': ms1, 'charge': charge,
                           'modifications': target_scan.get('modifications'), 'rt': rt}
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
                if len(finished) == len(precursors.keys()) and delta != -1:
                    break
                map_to_search = self.quant_mrm_map[mass] if self.mrm else self.quant_msn_map
                if current_scan is None:
                    current_scan = initial_scan
                else:
                    current_scan = find_prior_scan(map_to_search, current_scan) if delta == -1 else find_next_scan(map_to_search, current_scan)
                found = set([])
                if current_scan is not None:
                    if current_scan in scans_to_skip:
                        continue
                    else:
                        df = self.getScan(current_scan, start=None if self.mrm else precursor-5, end=None if self.mrm else precursor+highest_shift)
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
                            if precursor_label in finished:
                                continue
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
                                envelope = peaks.findEnvelope(xdata, ydata, measured_mz=measured_precursor, theo_mz=theoretical_precursor, max_mz=shift_max,
                                                              charge=charge, precursor_ppm=self.precursor_ppm, isotope_ppm=self.isotope_ppm, reporter_mode=self.reporter_mode,
                                                              isotope_ppms=self.isotope_ppms if self.fitting_run else None, quant_method=self.quant_method,
                                                              theo_dist=theo_dist if self.mono or precursor_shift == 0.0 else None, label=precursor_label, skip_isotopes=finished_isotopes[precursor_label],
                                                              last_precursor=last_precursors[delta].get(precursor_label, measured_precursor), isotopologue_limit=self.isotopologue_limit)
                                if not envelope['envelope']:
                                #    finished.add(precursor_label)
                                    continue
                                #if precursor_label == 'Medium':
                                 #   print df.name, envelope
                                if 0 in envelope['micro_envelopes'] and envelope['micro_envelopes'][0].get('int'):
                                    if ms_index == 0:
                                        last_precursors[delta*-1][precursor_label] = envelope['micro_envelopes'][0]['params'][1]
                                    last_precursors[delta][precursor_label] = envelope['micro_envelopes'][0]['params'][1]
                                added_keys = []
                                for isotope, vals in envelope['micro_envelopes'].iteritems():
                                    if isotope in finished_isotopes[precursor_label]:
                                        continue
                                    peak_intensity = vals.get('int')
                                    # if precursor_label == 'Medium':
                                    #     print peak_intensity, last_peak_height[precursor_label][isotope]
                                    # check the slope to see if we're just going off endlessly
                                    # if precursor_label == 'Light':
                                    #     print precursor_label, isotope, self.peak_cutoff, peak_intensity, low_int_isotopes[(precursor_label, isotope)], last_peak_height[precursor_label][isotope]
                                    if peak_intensity == 0 or (self.peak_cutoff and peak_intensity < last_peak_height[precursor_label][isotope]*self.peak_cutoff):
                                        low_int_isotopes[(precursor_label, isotope)] += 1
                                        if low_int_isotopes[(precursor_label, isotope)] >= 2:
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
                        if not self.mrm and len(labels_found) < self.labels_needed:
                            found.discard(precursor_label)
                            if df is not None and df.name in combined_data.columns:
                                del combined_data[df.name]
                                for i in isotopes_chosen.keys():
                                    if i[0] == df.name:
                                        del isotopes_chosen[i]
                        del df

                if not found or (np.abs(ms_index) > 7 and self.flat_slope(combined_data, delta)):
                    not_found += 1
                    # the 25 check is in case we're in something crazy. We should already have the elution profile of the ion
                    # of interest, else we're in an LC contaminant that will never end.
                    if current_scan is None or not_found >= 2:
                        not_found = 0
                        if delta == -1:
                            delta = 1
                            current_scan = initial_scan
                            finished = set([])
                            finished_isotopes = {i: set([]) for i in precursors.keys()}
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
                                    finished = set([])
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
            if isotopes_chosen and isotope_labels and not combined_data.empty:
                if self.mrm:
                    combined_data = combined_data.T
                # bookend with zeros if there aren't any, do the right end first because pandas will by default append there
                # if combined_data.iloc[:,-1].sum() != 0:
                combined_data = combined_data.sort(axis='index').sort(axis='columns')
                start_rt = rt
                if len(combined_data.columns) == 1:
                    try:
                        new_col = self.msn_rt_map.iloc[self.msn_rt_map.searchsorted(combined_data.columns[-1])+1].values[0]
                    except:
                        print combined_data.columns
                        print self.msn_rt_map
                else:
                    new_col = combined_data.columns[-1]+(combined_data.columns[-1]-combined_data.columns[-2])
                combined_data[new_col] = 0
                new_col = combined_data.columns[0]-(combined_data.columns[1]-combined_data.columns[0])
                combined_data[new_col] = 0
                combined_data = combined_data[sorted(combined_data.columns)]

                combined_data = combined_data.sort(axis='index').sort(axis='columns')
                quant_vals = defaultdict(dict)
                isotope_labels = pd.DataFrame(isotope_labels).T

                fig_map = {}

                isotopes_chosen = pd.DataFrame(isotopes_chosen).T
                isotopes_chosen.index.names = ['RT', 'MZ']
                label_fig_row = {v: i for i,v in enumerate(self.mrm_pair_info.columns)} if self.mrm else {v: i+1 for i,v in enumerate(precursors.keys())}

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
                        'common-x': ['x']+map(lambda x: '{0:0.2f}'.format(x), combined_data.columns),
                        'rows': len(precursors),
                        'max-y': combined_data.max().max(),
                    }
                    rt_figure_mapper = {}

                    for counter, (index, row) in enumerate(isotope_group):
                        try:
                            title = 'Scan {} RT {}'.format(self.msn_rt_map[self.msn_rt_map==index].index[0], index)
                        except:
                            title = '{}'.format(index)
                        # try:
                        #     isotope_figure['title'] =
                        # except:
                        #     pass
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
                    # fname = '{2}_{0}_{1}_{3}_clusters.png'.format(peptide, ms1, self.filename, scanId)
                    # subplot_rows = len(precursors.keys())+1
                    # subplot_columns = pd.Series(isotope_labels['label']).value_counts().iloc[0]+1
                    # fig = plt.figure(figsize=(subplot_columns*3 if subplot_columns*3 < 300 else 300, subplot_rows*4 if subplot_rows*4 < 300 else 300))
                    # combined_ax = fig.add_subplot(subplot_rows, subplot_columns, 1, projection='3d')
                    # for group, values in isotope_labels.groupby('label'):
                    #     ax = fig.add_subplot(subplot_rows, subplot_columns, label_fig_row.get(group)*subplot_columns+1, projection='3d')
                    #     for i in values.index:
                    #         Y = combined_data.loc[i].name.astype(float)
                    #         X = combined_data.loc[i].index.astype(float).values
                    #         Z = combined_data.loc[i].fillna(0).values
                    #         Xi,Yi = np.meshgrid(X, Y)
                    #         ax.plot_wireframe(Yi, Xi, Z, cmap=plt.cm.coolwarm)
                    #         combined_ax.plot_wireframe(Yi, Xi, Z, cmap=plt.cm.coolwarm)
                    #     plt.xticks(values.index.values, ['{0:0.2f}'.format(i) for i in values.index])
                    # if peptide:
                    #     plt.suptitle(peptide)
                    # elif mass:
                    #     plt.suptitle(mass)

                combined_peaks = defaultdict(dict)
                plot_index = {}
                fig_nums = defaultdict(list)
                for mz, label in isotope_labels['label'].iteritems():
                    fig_nums[label].append(mz)
                labelx = False
                labely = False

                merged_data = combined_data.sum(axis=0)
                merged_x = merged_data.index.astype(float).values
                merged_y = merged_data.values.astype(float)
                mval = merged_y.max()
                res, all_peaks = peaks.findAllPeaks(merged_x, merged_y, filter=True, bigauss_fit=True, rt_peak=start_rt)
                rt_means = res[1::4]
                rt_amps = res[::4]
                rt_std = res[2::4]
                rt_std2 = res[3::4]
                valid_peaks = [{'mean': i, 'amp': j*mval, 'std': l, 'std2': k, 'total': merged_y.sum()}
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
                valid_peaks.sort(key=lambda x: np.abs(x['mean']-start_rt))
                # print to_keep, to_remove
                # print valid_peak
                # print valid_peaks
                # print '---'
                #valid_peaks = valid_peaks if valid_peak is None else [valid_peak]
                peak_index = peaks.find_nearest_index(merged_x, valid_peaks[0]['mean'])
                peak_location = merged_x[peak_index]
                merged_lb = peaks.find_nearest_index(merged_x, valid_peaks[0]['mean']-valid_peaks[0]['std']*2)
                merged_rb = peaks.find_nearest_index(merged_x, valid_peaks[0]['mean']+valid_peaks[0]['std2']*2)
                merged_rb = len(merged_x) if merged_rb == -1 else merged_rb+1
                # print merged_x, merged_lb, merged_rb, valid_peaks, valid_peaks[0]['mean']-valid_peaks[0]['std']*2, valid_peaks[0]['mean']+valid_peaks[0]['std2']*2

                for row_num, (index, values) in enumerate(combined_data.iterrows()):
                    quant_label = isotope_labels.loc[index, 'label']
                    # if self.html:
                        # fig_index = label_fig_row.get(quant_label)*subplot_columns+fig_nums[quant_label].index(index)+2
                        # current_row = int(fig_index/subplot_columns+1)
                        # if (fig_index-2)%subplot_columns == 0:
                        #     labely = True
                        # if current_row == subplot_rows:
                        #     labelx = True
                        # fig_map[index] = fig_index
                        # plot_index[index, quant_label] = row_num

                    xdata = values.index.values.astype(float)
                    ydata = values.fillna(0).values.astype(float)
                    if sum(ydata>0) >= self.min_scans:
                        # print quant_label, index
                        #res, all_peaks = peaks.findAllPeaks(xdata, ydata, filter=True, bigauss_fit=True, rt_peak=peak_location)
                        peak_x = np.copy(xdata[merged_lb:merged_rb])
                        peak_y = np.copy(ydata[merged_lb:merged_rb])
                        if peak_x.size <= 1 or sum(peak_y>0) < self.min_scans:
                            continue
                        res = peaks.fixedMeanFit2(peak_x, peak_y, peak_index=peaks.find_nearest_index(peak_x, valid_peaks[0]['mean']))
                        # if quant_label == 'Light':
                        # print 'fmf', quant_label, index, res, peak_x, ydata[merged_lb:merged_rb], peaks.fixedMeanFit2(peak_x, np.copy(ydata[merged_lb:merged_rb]), peak_index=peaks.find_nearest_index(peak_x, valid_peaks[0]['mean']), debug=True)
                        #print peak_x
                        #print res, peaks.fixedMeanFit(peak_x, np.copy(ydata[merged_lb:merged_rb]), peak_index=peaks.find_nearest_index(peak_x, valid_peaks[0]['mean']))
                        if res is None:
                            continue
                        # res2, all_peaks2 = peaks.findAllPeaks2(values, filter=True)
                        # if len(res.x) > 4:
                        #     print res
                        # res2, all_peaks2 = peaks.findAllPeaks2(values, filter=True)
                        #print res1.success, res2.success, res1.fun, res2.fun, res1.bic, res2.bic
                        # res, all_peaks = (res, all_peaks) if res.bic < res2.bic else (res2, all_peaks2)
                        rt_means = res[1::4]
                        rt_amps = res[::4]
                        rt_std = res[2::4]
                        rt_std2 = res[3::4]
                        valid_peaks = [{'mean': i, 'amp': j, 'std': l, 'std2': k, 'total': values.sum()}
                                        for i, j, l, k in zip(rt_means, rt_amps, rt_std, rt_std2)]
                        # if we have a peaks containing our retention time, keep them and throw out ones not containing it
                        to_remove = []
                        to_keep = []
                        for i,v in enumerate(valid_peaks):
                            mu = v['mean']
                            s1 = v['std']
                            s2 = v['std2']
                            if mu-s1*2 < start_rt < mu+s2*2:
                                v['valid'] = True
                                to_keep.append(i)
                            elif peaks.find_nearest_index(merged_x, mu)-peak_location > 2:
                                to_remove.append(i)
                        # kick out peaks not containing our RT
                        valid_peak = None
                        if not to_keep:
                            # we have no peaks with our RT, there are contaminating peaks, remove all the noise but the closest to our RT
                            if not self.mrm:
                                # for i in to_remove:
                                #     valid_peaks[i]['interpolate'] = True
                                valid_peak = sorted([(i, np.abs(i['mean']-start_rt)) for i in valid_peaks], key=operator.itemgetter(1))[0][0]
                                for i in reversed(xrange(len(valid_peaks))):
                                    if valid_peaks[i] == valid_peak:
                                        continue
                                    else:
                                        del valid_peaks[i]
                                # valid_peak['interpolate'] = True
                            # else:
                            #     valid_peak = [j[0] for j in sorted([(i, i['amp']) for i in valid_peaks], key=operator.itemgetter(1), reverse=True)[:3]]
                        else:
                            for i in reversed(to_remove):
                                del valid_peaks[i]
                        combined_peaks[quant_label][index] = valid_peaks# if valid_peak is None else [valid_peak]

                    if self.html:
                        # ax = fig.add_subplot(subplot_rows, subplot_columns, fig_index)
                        if quant_label in rt_figure_mapper:
                            rt_base = rt_figure_mapper[(quant_label, index)]
                        else:
                            rt_base = {'data': {'x': 'x', 'columns': []}, 'grid': {'x': {'lines': [{'value': rt, 'text': 'Initial RT {0:0.2f}'.format(rt), 'position': 'middle'}]}}, 'subchart': {'show': True}, 'axis': {'x': {'label': 'Retention Time'}, 'y': {'label': 'Intensity'}}}
                            rt_figure_mapper[(quant_label, index)] = rt_base
                            rt_figure['data'].append(rt_base)
                        rt_base['data']['columns'].append(['{0} {1} raw'.format(quant_label, index)]+ydata.tolist())

                peak_info = {i: {'amp': -1, 'var': 0} for i in self.mrm_pair_info.columns} if self.mrm else {i: {'amp': -1, 'var': 0} for i in data.keys()}
                if combined_peaks:
                    common_peak = self.replaceOutliers(combined_peaks, combined_data)
                    common_loc = peaks.find_nearest_index(xdata, common_peak)#np.where(xdata==common_peak)[0][0]

                    for quant_label, quan_values in combined_peaks.items():
                        for index, values in quan_values.items():
                            if not values:
                                continue
                            rt_values = combined_data.loc[index]
                            xdata = rt_values.index.values.astype(float)
                            ydata = rt_values.fillna(0).values.astype(float)
                            # pick the biggest within a rt cutoff of 0.2, otherwise pick closest
                            # closest_rts = sorted([(i, i['amp']) for i in values if np.abs(i['peak']-common_peak) < 0.2], key=operator.itemgetter(1), reverse=True)
                            # if not closest_rts:
                            # print values
                            closest_rts = sorted([(i, np.abs(i['mean']-common_peak)) for i in values], key=operator.itemgetter(1))
                            # print closest_rts
                            closest_rt = closest_rts[0][0]
                            # if we move more than a # of ms1 to the dominant peak, update to our known peak
                            gc = 'k'
                            pos_x = xdata[ydata>0]
                            nearest = peaks.find_nearest_index(pos_x, closest_rt['mean'])
                            peak_loc = np.where(xdata==pos_x[nearest])[0][0]
                            mean = closest_rt['mean']
                            amp = closest_rt['amp']
                            mean_diff = mean-xdata[common_loc]
                            mean_diff = np.abs(mean_diff/closest_rt['std'] if mean_diff < 0 else mean_diff/closest_rt['std2'])
                            std = closest_rt['std']
                            std2 = closest_rt['std2']
                            if False and len(xdata) >= 3 and (mean_diff > 2 or (np.abs(peak_loc-common_loc) > 2 and mean_diff > 2)):
                                # fixed mean fit
                                if self.debug:
                                    print quant_label, index
                                    print common_loc, peak_loc
                                nearest = peaks.find_nearest_index(pos_x, mean)
                                nearest_index = np.where(xdata==pos_x[nearest])[0][0]
                                res = peaks.fixedMeanFit(xdata, ydata, peak_index=nearest_index, debug=self.debug)
                                if res is None:
                                    continue
                                amp, mean, std, std2 = res
                                amp *= ydata.max()
                                gc = 'g'
                            #var_rat = closest_rt['var']/common_var
                            peak_params = np.array([amp,  mean, std, std2])
                            # int_args = (res.x[rt_index]*mval, res.x[rt_index+1], res.x[rt_index+2])
                            left, right = xdata[0]-4*std, xdata[-1]+4*std2
                            xr = np.linspace(left, right, 1000)
                            try:
                                int_val = integrate.simps(peaks.bigauss_ndim(xr, peak_params), x=xr) if self.quant_method == 'integrate' else ydata[(xdata > left) & (xdata < right)].sum()
                            except:
                                print traceback.format_exc()
                                print xr, peak_params
                            isotope_index = isotope_labels.loc[index, 'isotope_index']

                            if int_val and not pd.isnull(int_val) and gc != 'c':
                                try:
                                    quant_vals[quant_label][isotope_index] += int_val
                                except KeyError:
                                    quant_vals[quant_label][isotope_index] = int_val
                            if peak_info.get(quant_label, {}).get('amp', -1) < amp:
                                peak_info[quant_label].update({'amp': amp, 'std': std, 'std2': std2, 'mean_diff': mean_diff})
                            if self.html:
                                rt_base = rt_figure_mapper[(quant_label, index)]
                                key = '{} {}'.format(quant_label, index)
                                for i,v in enumerate(rt_base['data']['columns']):
                                    if key in v[0]:
                                        break
                                rt_base['data']['columns'].insert(i, ['{0} {1} fit'.format(quant_label, index)]+np.nan_to_num(peaks.bigauss_ndim(xdata, peak_params)).tolist())
                                pass
                                # ax = fig.add_subplot(subplot_rows, subplot_columns, fig_map.get(index))
                                # try:
                                #     ax.set_title('{0:0.2f} AUC: {1}'.format(index, int(int_val)))
                                # except:
                                #     ax.set_title('{0:0.2f} AUC: {1}'.format(index, 'NA'))
                                # plot_points = np.linspace(xdata[0], xdata[-1], 100)
                                # if self.quant_method == 'integrate':
                                #     ax.plot(plot_points, peaks.bigauss_ndim(plot_points, peak_params), '{}o-'.format(gc), alpha=0.7)
                                # else:
                                #     ax.bar(xdata, ydata, width=((xdata[1]-xdata[0]) if len(xdata) > 1 else 1)/4, alpha=0.7)
                                # ax.plot([start_rt, start_rt], ax.get_ylim(),'k-')
                                # ax.set_ylim(0,combined_data.max().max())
                                # x_for_plot = xdata[::int(len(xdata)/5)+1]
                                # ax.set_xlim(xdata[0], xdata[-1])
                                # ax.set_xticks(x_for_plot)
                                # ax.set_xticklabels(['{0:.2f} '.format(i) for i in x_for_plot], rotation=45, ha='right')
                write_html = True if self.ratio_cutoff == 0 else False
                for silac_label1 in data.keys():
                    qv1 = quant_vals.get(silac_label1)
                    for silac_label2 in data.keys():
                        if silac_label1 == silac_label2:
                            continue
                        qv2 = quant_vals.get(silac_label2)
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

                if self.html:
                    pass
                    # plt.tight_layout()
                    # ax.get_figure().savefig(os.path.join(self.html['full'], fname), format='png', dpi=100)
                    # html_images['clusters'] = os.path.join(self.html['rel'], fname)
                if self.debug or self.html:
                    plt.close('all')
                if write_html:
                    result_dict.update({'html_info': html_images})
                for silac_label, silac_data in data.iteritems():
                    w1 = peak_info.get(silac_label, {}).get('std', None)
                    w2 = peak_info.get(silac_label, {}).get('std2', None)
                    result_dict.update({
                        '{}_intensity'.format(silac_label): sum(quant_vals[silac_label].values()),
                        '{}_peak_intensity'.format(silac_label): peak_info.get(silac_label, {}).get('amp', 'NA'),
                        '{}_isotopes'.format(silac_label): sum(isotopes_chosen['label'] == silac_label),
                        '{}_rt_width'.format(silac_label): w1+w2 if w1 and w2 else 'NA',
                        '{}_mean_diff'.format(silac_label): peak_info.get(silac_label, {}).get('mean_diff', 'NA'),

                    })
                del combined_peaks
            for silac_label, silac_data in data.iteritems():
                result_dict.update({
                    '{}_precursor'.format(silac_label): silac_data['precursor'],
                    '{}_calibrated_precursor'.format(silac_label): silac_data.get('calibrated_precursor', silac_data['precursor'])
                })
            result_dict.update({
                'ions_found': target_scan.get('ions_found'),
                'html': {'peptide': rt_figure, 'rt': isotope_figure}
            })
            self.results.put(result_dict)
            del result_dict
            del data
            del combined_data
            del isotopes_chosen
        except:
            sys.stderr.write('ERROR ON {}'.format(traceback.format_exc()))
            return

    def run(self):
        for index, params in enumerate(iter(self.queue.get, None)):
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
        if scan_found is True:
            if ms_level is None:
                return scan_id
            elif scan_msn == ms_level:
                return scan_id
        if scan_found is False and scan_id == current_scan:
            scan_found = True
    return None

def find_scan(msn_map, current_scan):
    for scan_msn, scan_id in msn_map:
        if scan_id == current_scan:
            return scan_id
    return None

def main():
    args = parser.parse_args()
    isotopologue_limit = args.isotopologue_limit
    isotopologue_limit = isotopologue_limit if isotopologue_limit else None
    labels_needed = args.labels_needed
    overlapping_mz = args.overlapping_mz
    threads = args.p
    skip = args.skip
    out = args.out
    html = args.html
    resume = args.resume
    calc_stats = not args.disable_stats
    msn_for_id = args.msn
    raw_data_only = not (args.processed or args.tsv)
    msn_for_quant = args.msn_quant_from if args.msn_quant_from else msn_for_id-1
    if msn_for_quant == 0:
        msn_for_quant = 1
    reporter_mode = args.reporter_ion
    msn_ppm = args.msn_ppm
    quant_method = args.quant_method
    if msn_for_quant == 1 and quant_method is None:
        quant_method = 'integrate'
    elif msn_for_quant > 1 and quant_method is None:
        quant_method = 'sum'

    mrm_pair_info = pd.read_table(args.mrm_map) if args.mrm and args.mrm_map else None

    scan_filemap = {}
    found_scans = {}
    raw_files = {}
    silac_labels = {'Light': {0: set([])}} if not reporter_mode else {}

    name_mapping = {}

    if args.label_scheme:
        silac_labels = {}
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
            silac_labels.update({label_name: masses})
    if args.label_method:
        silac_labels = config.MS1_SCHEMES[args.label_method]

    sample = args.sample
    sys.stderr.write('Loading Scans:\n')

    # options determining modes to quantify
    all_msn = False # we just have a raw file
    ion_search = False # we have an ion we want to find

    input_found = None
    if args.processed:
        results = GuessIterator(args.processed.name, full=True, store=False, peptide=args.peptide)
        input_found = 'ms'
    elif args.tsv:
        results = pd.read_table(args.tsv, sep='\t')
        input_found = 'tsv'

    if args.processed:
        source_file = args.processed.name
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
            if args.peptide and not any([j.lower() == peptide.lower() for j in args.peptide]):
                continue
            if not args.peptide and (sample != 1.0 and random.random() > sample):
                continue
            specId = str(i[scan_col])
            fname = i[file_col] if file_col in i else raw_file
            if fname not in scan_filemap:
                fname = os.path.split(fname)[1]
                if fname not in scan_filemap:
                    if skip:
                        continue
                    sys.stderr.write('{0} not found in filemap. Filemap is {1}. If you wish to ignore this message, add --skip to your input arguments.'.format(fname, scan_filemap))
                    return 1
            mass_key = (specId, fname)
            if mass_key in found_scans:
                continue
            charge = float(i[charge_col]) if charge_col in i else 1
            precursor_mass = i[precursor_col] if precursor_col in i else None
            rt_value = i[rt_col] if rt_col in i else None
            #'id': id_Scan[id], 'theor_mass' -> id_scan[mass], 'peptide': idScan[peptide,],  'mod_peptide': idscan, 'rt': idscan,
            #
            d = {
                'file': fname, 'quant_scan': {}, 'id_scan': {
                'id': specId, 'mass': precursor_mass, 'peptide': peptide, 'rt': rt_value,
                'charge': charge, 'modifications': None, 'label': name_mapping.get(i[label_col]) if label_col in i else None
            }
            }
            found_scans[mass_key] = d
            if args.replicate:
                for i in scan_filemap:
                    raw_files[i] = d
            else:
                try:
                    raw_files[i[file_col]].append(d)
                except:
                    raw_files[i[file_col]] = [d]
    elif input_found == 'ms':
        if not (args.label_scheme or args.label_method):
            silac_labels.update(results.getSILACLabels())
        scans_to_select = set(args.scan if args.scan else [])
        replicate_file_mapper = {}
        for index, scan in enumerate(results.getScans(modifications=False, fdr=True)):
            if index%1000 == 0:
                sys.stderr.write('.')
            if scan is None:
                continue
            peptide = scan.peptide
            # if peptide.lower() != 'AGkPVIcATQMLESmIk'.lower():
            #     continue
            if args.peptide and not any([j.lower() == peptide.lower() for j in args.peptide]):
                continue
            if not args.peptide and (sample != 1.0 and random.random() > sample):
                continue
            specId = scan.id
            if args.scan and specId not in scans_to_select:
                continue
            fname = scan.file
            mass_key = (fname, specId, peptide)
            if mass_key in found_scans:
                continue
            d = {
                'file': fname, 'quant_scan': {}, 'id_scan': {
                'id': specId, 'theor_mass': scan.getTheorMass(), 'peptide': peptide, 'mod_peptide': scan.modifiedPeptide, 'rt': scan.rt,
                'charge': scan.charge, 'modifications': scan.getModifications(), 'mass': float(scan.mass)
            }
            }
            found_scans[mass_key] = d#.add(mass_key)
            fname = os.path.splitext(fname)[0]
            if args.replicate:
                # find the most similar name, add in a setting for this
                import difflib
                if fname in replicate_file_mapper:
                    fname = replicate_file_mapper[fname]
                else:
                    s = difflib.SequenceMatcher(None, fname)
                    seq_matches = []
                    for i in scan_filemap:
                        s.set_seq2(i)
                        seq_matches.append((s.ratio(), i))
                    seq_matches.sort(key=operator.itemgetter(1), reverse=True)
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
    if scan_filemap and raw_data_only:
        # determine if we want to do ms1 ion detection, ms2 ion detection, all ms2 of each file
        if args.msn_ion or args.msn_peaklist:
            RESULT_ORDER.extend([('ions_found', 'Ions Found')])
            ion_search = True
            ions_selected = args.msn_ion if args.msn_ion else [float(i.strip()) for i in args.msn_peaklist if i]
            d = {'ions': ions_selected}
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

    labels = silac_labels.keys()
    for silac_label in labels:
        RESULT_ORDER.extend([('{}_intensity'.format(silac_label), '{} Intensity'.format(silac_label)),
                             ('{}_precursor'.format(silac_label), '{} Precursor'.format(silac_label)),
                             ('{}_calibrated_precursor'.format(silac_label), '{} Calibrated Precursor'.format(silac_label)),
                             ('{}_rt_width'.format(silac_label), '{} RT Width'.format(silac_label)),
                             ('{}_isotopes'.format(silac_label), '{} Isotopes Found'.format(silac_label)),
                             ('{}_mean_diff'.format(silac_label), '{} Mean Offset'.format(silac_label)),
                             ('{}_peak_intensity'.format(silac_label), '{} Peak Intensity'.format(silac_label)),
                             ])
        for silac_label2 in labels:
            if silac_label != silac_label2:
                RESULT_ORDER.extend([('{}_{}_ratio'.format(silac_label, silac_label2), '{}/{}'.format(silac_label, silac_label2)),
                                     ])
                if calc_stats:
                    RESULT_ORDER.extend([('{}_{}_confidence'.format(silac_label, silac_label2), '{}/{} Confidence'.format(silac_label, silac_label2)),
                                         ])

    pq_dir = os.path.split(__file__)[0]

    workers = []
    completed = 0
    sys.stderr.write('Beginning quantification.\n')
    scan_count = len(found_scans)
    headers = ['Raw File']+[i[1] for i in RESULT_ORDER]
    if resume and os.path.exists(out):
        if not out:
            sys.stderr.write('You may only resume runs with a file output.\n')
            return -1
        out = open(out, 'ab')
        out_path = out.name
    else:
        if out:
            out = open(out, 'wb')
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
            out = []
            html_output = {}
            for col_index, (i,v) in enumerate(zip(l.split('\t'), keys)):
                if v in html_extra:
                    html_output[col_index] = html_extra[v]
                out.append('<td>{0}</td>'.format(i))
            res += '\n'.join(out)+'</tr>'
            return table_rows(html_list, res=res), html_output

        if resume:
            html_out = open('{0}.html'.format(out_path), 'ab')
        else:
            html_out = open('{0}.html'.format(out_path), 'wb')
            template = []
            for i in open(os.path.join(pq_dir, 'pyquant_output.html'), 'rb'):
                if 'HTML BREAK' in i:
                    break
                template.append(i)
            html_template = Template(''.join(template))
            html_out.write(html_template.substitute({'title': source_file, 'table_header': '\n'.join(['<th>{0}</th>'.format(i) for i in ['Raw File']+[i[1] for i in RESULT_ORDER]])}))

    skip_map = set([])
    if resume:
        with open('{}.tmp'.format(out.name), 'rb') as temp_file:
            for index, entry in enumerate(temp_file):
                info = json.loads(entry)['res_list']
                # key is filename, peptide, charge, target scan id, modifications
                key = (info[0], info[1], info[3], info[5], info[2])
                skip_map.add(key)
        temp_file = open('{}.tmp'.format(out.name), 'ab')
    else:
        temp_file = open('{}.tmp'.format(out.name), 'wb')

    silac_shifts = {}
    for silac_label, silac_masses in silac_labels.items():
        for mass, aas in silac_masses.iteritems():
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
        reader_outs = {}
        for i in xrange(threads):
            reader_outs[i] = Queue()

        msn_map = []
        scan_rt_map = {}
        msn_rt_map = {}
        scan_charge_map = {}

        raw = GuessIterator(filepath, full=False, store=False)
        sys.stderr.write('Processing {}.\n'.format(filepath))

        # params in case we are doing ion search or replicate analysis
        ion_tolerance = args.precursor_ppm/1e6 if args.replicate else args.msn_ppm/1e6
        ion_search_list = []
        scans_to_fetch = []

        # figure out the splines for mass accuracy correction
        calc_spline = args.no_mass_accuracy_correction is False and raw_data_only is False
        spline_x = []
        spline_y = []
        spline = None

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
            scan_charge_map[scan_id] = scan.charge
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
            elif args.replicate:
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

        reader = Reader(reader_in, reader_outs, raw_file=raw, spline=spline)
        reader.start()
        rep_map = defaultdict(set)
        if ion_search or args.replicate:
            ions = [i['id_scan'].get('theor_mass', i['id_scan']['mass']) for i in raw_scans] if args.replicate else raw_scans['ions']
            last_scan_ions = set([])
            for scan_id in scans_to_fetch:
                reader_in.put((0, scan_id, None, None))
                scan = reader_outs[0].get()
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
                        if args.replicate:
                            d = raw_scans[ion_index]
                            scan_rt = d['id_scan']['rt']
                            if scan_rt-args.rt_window < rt < scan_rt+args.rt_window:
                                ions_found.append({'ion': ion, 'nearest_mz': nearest_mz, 'charge': d['id_scan']['charge'], 'scan_info': d})
                        else:
                            ions_found.append({'ion': ion, 'nearest_mz': nearest_mz})
                if ions_found:
                    # we have two options here. If we are quantifying a preceeding scan or the ion itself per scan
                    isotope_ppm = args.isotope_ppm/1e6
                    if msn_for_quant == msn_for_id or args.replicate:
                        for ion_dict in ions_found:
                            ion, nearest_mz = ion_dict['ion'], ion_dict['nearest_mz']
                            if not reporter_mode and ion in last_scan_ions:
                                continue
                            ion_found = '{}({})'.format(ion, nearest_mz)
                            spectra_to_quant = scan_id
                            # we are quantifying the ion itself
                            if charge == 0 or args.replicate:
                                # see if we can figure out the charge state
                                charge_states = []
                                for i in xrange(1,5):
                                    to_add = True
                                    peak_height = 0
                                    for j in xrange(1,3):
                                        next_peak = ion+peaks.NEUTRON/float(i)*j
                                        closest_mz = peaks.find_nearest_index(scan_mzs, next_peak)
                                        if peaks.get_ppm(next_peak, scan_mzs[closest_mz]) > isotope_ppm*1.5:
                                            to_add = False
                                        peak_height += mz_vals[closest_mz]
                                    if to_add:
                                        charge_states.append((i, peak_height))
                                # print int(ion_dict['charge']), charge_states
                                # print int(ion_dict['charge']), charge_states
                                if args.replicate and int(ion_dict['charge']) not in [i[0] for i in charge_states]:
                                    continue
                                elif args.replicate:
                                    charge_to_use = ion_dict['charge']
                                else:
                                    charge_to_use = sorted(charge_states, key=operator.itemgetter(1), reverse=True)[0][0] if charge_states else 1
                                rep_key = (ion_dict['scan_info']['id_scan']['rt'], ion_dict['scan_info']['id_scan']['mass'], charge_to_use)
                                rep_map[rep_key].add(rt)
                            else:
                                charge_to_use = charge
                            if args.replicate:
                                d = copy.deepcopy(ion_dict['scan_info'])
                                d['id_scan']['id'] = scan_id
                                spectra_to_quant = find_prior_scan(msn_map, scan_id, ms_level=msn_for_quant)
                                d['quant_scan']['id'] = scan_id
                            else:
                                d = {
                                    'quant_scan': {'id': scan_id},
                                    'id_scan': {
                                        'id': scan_id, 'theor_mass': ion, 'rt': rt,
                                        'charge': charge_to_use, 'mass': float(nearest_mz), 'ions_found': ion_found,
                                    },
                                }
                            key = (scan_id, d['id_scan']['theor_mass'], charge_to_use)
                            if key in added:
                                continue
                            added.add(key)
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
                last_scan_ions = set([i['ion'] for i in ions_found])
                del scan

        if args.replicate:
            x = []
            y = []
            for i,v in rep_map.iteritems():
                for j in v:
                    x.append(i[0])
                    y.append(j)
            from sklearn.linear_model import LinearRegression
            rep_mapper = LinearRegression()
            rep_mapper.fit(np.array(x).reshape(len(x), 1), y)

        if ion_search or all_msn or args.replicate:
            scan_count = len(ion_search_list)
            raw_scans = [i[1] for i in sorted(ion_search_list, key=operator.itemgetter(0))]

        for i in xrange(threads):
            worker = Worker(queue=in_queue, results=result_queue, raw_name=filepath, silac_labels=silac_labels,
                            debug=args.debug, html=html, mono=not args.spread, precursor_ppm=args.precursor_ppm,
                            isotope_ppm=args.isotope_ppm, isotope_ppms=None, msn_rt_map=msn_rt_map, reporter_mode=reporter_mode,
                            reader_in=reader_in, reader_out=reader_outs[i], thread=i, quant_method=quant_method,
                            spline=spline, isotopologue_limit=isotopologue_limit, labels_needed=labels_needed,
                            quant_msn_map=filter(lambda x: x[0] == msn_for_quant, msn_map) if not args.mrm else msn_map,
                            overlapping_mz=overlapping_mz, min_resolution=args.min_resolution, min_scans=args.min_scans,
                            mrm_pair_info=mrm_pair_info, mrm=args.mrm, peak_cutoff=args.peak_cutoff, replicate=args.replicate)
            workers.append(worker)
            worker.start()

        # TODO:
        # combine information from scans (where for instance, we have fragmented both the heavy/light
        # peptides -- we want to use those masses before calculating where it should be). This may not
        # be possible for all types of input though, figure this out.
        quant_map = {}
        msn_rt_map_series = pd.Series(msn_rt_map)
        scans_to_submit = []

        lowest_label = min([j for i,v in silac_labels.items() for j in v])
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
                # figure out the ms-1 from the ms level we are at
                # find the closest scan to this, which will be the parent scan
                msn_to_quant = find_prior_scan(msn_map, scanId, ms_level=msn_for_quant)
                quant_scan['id'] = msn_to_quant

            rt = target_scan.get('rt', scan_rt_map.get(scanId))
            if rt is None:
                rt = float(msn_rt_map[msn_to_quant])
                target_scan['rt'] = rt

            if args.replicate:
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
                target_scan['precursor'] = target_scan['mass']-mass_shift
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
        map(in_queue.put, [i[1] for i in scans_to_submit])

        sys.stderr.write('{0} processed and placed into queue.\n'.format(filename))

        # kill the workers
        [in_queue.put(None) for i in xrange(threads)]
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
                        to_del.append(i)
                for i in sorted(to_del, reverse=True):
                    del workers[i]
            if result is not None:
                completed += 1
                if completed % 10 == 0:
                    sys.stderr.write('\r{0:2.2f}% Completed'.format(completed/scan_count*100))
                    sys.stderr.flush()
                res_list = [filename]+[result.get(i[0], 'NA') for i in RESULT_ORDER]
                key_order = [False]+[i[0] if i[0] in result else False for i in RESULT_ORDER]
                if calc_stats:
                    temp_file.write(json.dumps({'res_list': res_list, 'html': result.get('html', {})}))
                    temp_file.write('\n')
                    temp_file.flush()
                res = '{0}\n'.format('\t'.join(map(str, res_list)))
                out.write(res)
                out.flush()
                if html and not calc_stats:
                    html_out.write(table_rows([{'table': res.strip(), 'keys': key_order, 'html': result.get('html', {})}]))
                    html_out.flush()
        reader_in.put(None)
        del msn_map
        del scan_rt_map
        del raw

    temp_file.close()
    df_data = []
    html_data = []
    for j in open(temp_file.name, 'rb'):
        result = json.loads(j)
        df_data.append(result['res_list'])
        html_data.append(result['html'])
    data = pd.DataFrame.from_records(df_data, columns=[i for i in headers if i != 'Confidence'])
    if calc_stats:
        from scipy import stats
        header_mapping = []
        order_names = [i[1] for i in RESULT_ORDER]
        for i in data.columns:
            try:
                header_mapping.append(RESULT_ORDER[order_names.index(i)][0])
            except ValueError:
                header_mapping.append(False)
        data = data.replace('NA', np.nan)
        for silac_label1 in silac_labels.keys():
            label1_log = 'L{}'.format(silac_label1)
            label1_logp = 'L{}_p'.format(silac_label1)
            label1_int = '{} Intensity'.format(silac_label1)
            label1_hif = '{} Isotopes Found'.format(silac_label1)
            label1_hifp = '{} Isotopes Found p'.format(silac_label1)
            for silac_label2 in silac_labels.keys():
                if silac_label1 == silac_label2:
                    continue
                label2_log = 'L{}'.format(silac_label2)
                label2_logp = 'L{}_p'.format(silac_label2)
                label2_int = '{} Intensity'.format(silac_label2)
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
                data[mixed_confidence] = 10
                data.loc[(data[mixed_mean_p] > 0.90), mixed_confidence] -= 1
                data.loc[(data[mixed_rt_diff_p] > 0.90), mixed_confidence] -= 1
                data.loc[(data[label2_logp] < 0.10), mixed_confidence] -= 0.5
                data.loc[(data[label1_logp] < 0.10), mixed_confidence] -= 0.5
                data.loc[(data[mixed_p] < 0.10), mixed_confidence] -= 0.5
                data.loc[((data[mixed_isotope_diff_p] > 0.90) | (data[mixed_isotope_diff_p] < 0.10)), mixed_confidence] -= 1
                data.loc[(data[label2_hifp] < 0.10), mixed_confidence] -= 1
                data.loc[(data[label2_hifp] < 0.10), mixed_confidence] -= 1

        data.to_csv('{}_stats'.format(out.name), sep=str('\t'), index=None)

    out.flush()
    out.close()

    if html:
        html_map = []
        for index, (row_index, row) in enumerate(data.iterrows()):
            res = '\t'.join(row.astype(str))
            to_write, html_info = table_rows([{'table': res.strip(), 'html': html_data[index], 'keys': header_mapping}])
            html_map.append(html_info)
            html_out.write(to_write)
            html_out.flush()

        template = []
        append = False
        for i in open(os.path.join(pq_dir, 'pyquant_output.html'), 'rb'):
            if 'HTML BREAK' in i:
                append = True
            elif append:
                template.append(i)
        html_template = Template(''.join(template))
        html_out.write(html_template.safe_substitute({'html_output': base64.b64encode(gzip.zlib.compress(json.dumps(html_map), 9))}))

    os.remove(temp_file.name)

if __name__ == "__main__":
    sys.exit(main())