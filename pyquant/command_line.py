from __future__ import division, unicode_literals, print_function
import base64
import copy
import gzip
import os
import operator
import traceback
import random
import signal
import sys
from collections import defaultdict, OrderedDict
from functools import partial
from multiprocessing import Queue
from string import Template

import pandas as pd
import six
from pythomics.proteomics.parsers import GuessIterator
from pythomics.proteomics import config
from scipy.interpolate import UnivariateSpline
from six.moves.queue import Empty
from six.moves import xrange
try:
    from profilestats import profile
    from memory_profiler import profile as memory_profiler
except ImportError:
    pass


from .reader import Reader
from .worker import Worker
from .utils import find_prior_scan, get_scans_under_peaks, naninfmean, naninfsum, perform_ml
from . import peaks


description = """
PyQuant is a quantification program for mass spectrometry data. It attempts to be a general implementation to quantify
an assortment of datatypes and allows a high degree of customization for how data is to be quantified.
"""

ION_CUTOFF = 2

CRASH_SIGNALS = {signal.SIGSEGV, }

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
    if args.gcms:
        msn_for_quant = 1
        msn_for_id = 1
        args.precursor_ppm = args.msn_ppm
        isotopologue_limit = 1
        args.msn_all_scans = True
        args.no_mass_accuracy_correction = True
        args.no_contaminant_detection = True
        args.no_rt_guide = True
        args.disable_stats = True

    mrm_pair_info = pd.read_table(args.mrm_map) if args.mrm and args.mrm_map else None

    scan_filemap = {}
    found_scans = {}
    raw_files = {}
    mass_labels = {}#{'Light': {0: set([])}} if not reporter_mode else {}

    name_mapping = {}

    if args.label_scheme:
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
    elif args.scan_file_dir:
        source_file = os.path.split(args.scan_file_dir)[1]

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

    if scan_filemap and raw_data_only:
        # determine if we want to do ms1 ion detection, ms2 ion detection, all ms2 of each file
        if args.msn_ion or args.msn_peaklist:
            ion_search = True
            ions_of_interest = []
            if args.msn_peaklist:
                for i in args.msn_peaklist:
                    if not i.strip():
                        continue
                    try:
                        ions_of_interest.append(i.strip())
                    except:
                        import traceback; traceback.print_exc()
            if args.msn_ion:
                ions_of_interest += args.msn_ion
            ions_selected = [sorted(list(set(map(float, ion.split(','))))) for ion in ions_of_interest] if ions_of_interest else []
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

    # before we start, get the information we will be generating
    labels = mass_labels.keys()
    if not labels:
        if ion_search:
            if args.require_all_ions:
                labels = ['_'.join(map(str, ion_set)) for ion_set in ions_selected]
            else:
                labels = list(map(str, set([ion for ion_set in ions_selected for ion in ion_set])))
        else:
            labels = ['']
    if not scan_filemap and raw_data_only:
        RESULT_ORDER = []
    else:
        RESULT_ORDER = [
            ('peptide', 'Peptide'),
            ('modifications', 'Modifications'),
            ('accession', 'Accession'),
        ]
    RESULT_ORDER.extend([
        ('charge', 'Charge'),
        ('ms1', 'MS{} Spectrum ID'.format(msn_for_quant)),
    ])
    if msn_for_quant != msn_for_id:
        RESULT_ORDER.extend([
            ('scan', 'MS{} Spectrum ID'.format(msn_for_id)),
        ])
    RESULT_ORDER.extend([
        ('rt', 'Retention Time'),
    ])

    if ion_search and not args.msn_all_scans:
        RESULT_ORDER.extend([('ions_found', 'Ions Found')])

    PEAK_REPORTING = []
    if not reporter_mode and args.peaks_n != 1:
        PEAK_REPORTING.extend([
            ('label', 'Peak Label'),
            ('auc', 'Peak Area'),
            ('amp', 'Peak Max'),
            ('mean', 'Peak Center'),
            ('peak_width', 'RT Width'),
            ('mean_diff', 'Mean Offset'),
            ('snr', 'SNR'),
            ('sbr', 'SBR'),
            ('sdr', 'Density'),
            ('coef_det', 'R^2'),
            ('residual', 'Residual'),
        ])
    iterator = ['_'.join(map(str, sorted(labels)))] if args.merge_labels else labels
    for label in iterator:
        label_key = '{} '.format(label) if label != '' else label
        RESULT_ORDER.extend([
            ('{}_precursor'.format(label), '{}Precursor'.format(label_key)),
        ])
        if msn_for_quant == 1:
            if not args.no_mass_accuracy_correction:
                RESULT_ORDER.extend([
                    ('{}_calibrated_precursor'.format(label), '{}Calibrated Precursor'.format(label_key)),
                ])
            RESULT_ORDER.extend([
                ('{}_isotopes'.format(label), '{}Isotopes Found'.format(label_key)),
            ])
        RESULT_ORDER.extend([
            ('{}_intensity'.format(label), '{}Intensity'.format(label_key), partial(naninfsum, empty=0)),
        ])
        if args.peaks_n == 1:
            RESULT_ORDER.extend([
                ('{}_peak_width'.format(label), '{}RT Width'.format(label_key), partial(naninfmean, empty=0)),
                ('{}_mean_diff'.format(label), '{}Mean Offset'.format(label_key), partial(naninfmean, empty=np.NaN)),
            ])

        if msn_for_quant == 1:
            RESULT_ORDER.extend([
                ('{}_residual'.format(label), '{}Residual'.format(label_key), partial(naninfsum, empty=np.NaN)),
                ('{}_coef_det'.format(label), '{}R^2'.format(label_key), partial(naninfmean, empty=0)),
                ('{}_snr'.format(label), '{}SNR'.format(label_key), partial(naninfmean, empty=0)),
            ])
        if not args.merge_labels and not args.no_ratios and label != '':
            for label2 in labels:
                label_key2 = '{}'.format(label2) if label2 != '' else label2
                if label_key != label_key2 and (ref_label is None or ref_label.lower() == label_key2.lower()):
                    RESULT_ORDER.extend([('{}_{}_ratio'.format(label, label2), '{}/{}'.format(label_key.strip(), label_key2.strip())),
                                         ])
                    if calc_stats:
                        RESULT_ORDER.extend([('{}_{}_confidence'.format(label, label2), '{}/{} Confidence'.format(label_key.strip(), label_key2.strip())),
                                             ])

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
                    info = pd.io.json.loads(entry)['res_list']
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
                    # if not args.require_all_ions:
                    #     ions = set([j for i in ions for j in i])
                    for ion_index, ion_set in enumerate(ions):
                        d = {
                            'quant_scan': {'id': scan_id, 'scans': scans_to_fetch},
                            'id_scan': {
                                'id': scan_id, 'theor_mass': ion_set[0], 'rt': rt_info[ion_index] if rt_info else scan['rt'],
                                'charge': 1, 'mass': ion_set[0], 'ions_found': None, 'ion_set': ion_set,
                            },
                            'combine_xics': args.require_all_ions,
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
                for ion_set in ions:
                    for ion_index, (ion, nearest_mz_index) in enumerate(zip(ion_set, peaks.find_nearest_indices(scan_mzs, np.array(ion_set, dtype=np.float)))):
                        nearest_mz = scan_mzs[nearest_mz_index]
                        found_ions = []
                        if peaks.get_ppm(ion, nearest_mz) < ion_tolerance:
                            if args.mva:
                                d = raw_scans[ion_index]
                                scan_rt = d['id_scan']['rt']
                                if scan_rt-args.rt_window < rt < scan_rt+args.rt_window:
                                    found_ions.append({
                                        'ion': ion,
                                        'nearest_mz': nearest_mz,
                                        'charge': d['id_scan']['charge'],
                                        'scan_info': d,
                                        'ion_set': ion_set,
                                    })
                            else:
                                ion_rt = rt_info[ion_index] if rt_info else None
                                if ion_rt is None or (rt-args.rt_window < ion_rt < rt+args.rt_window):
                                    found_ions.append({
                                        'ion': ion,
                                        'nearest_mz': nearest_mz,
                                        'rt': ion_rt,
                                        'ion_set': ion_set,
                                    })
                        elif args.require_all_ions:
                            # this means an ion in the set was not found, if we require all ions, continue to the next set
                            continue
                        # if we require all ions, only put in a single copy since we'll be finding the same ions in every
                        # scan anyways, otherwise put the entire set in since we may be missing ions
                        ions_found += found_ions[0] if args.require_all_ions else found_ions
                if ions_found:
                    # we have two options here. If we are quantifying a preceeding scan or the ion itself per scan
                    isotope_ppm = args.isotope_ppm/1e6
                    if msn_for_quant == msn_for_id or args.mva:
                        for ion_dict in ions_found:
                            ion, nearest_mz = ion_dict['ion'], ion_dict['nearest_mz']
                            ion_found = ','.join(map(str, ion_dict['ion_set']))
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
                                        'ion_set': ion_dict['ion_set']
                                    },
                                    'combine_xics': True,
                                }
                            key = (scan_id, d['id_scan']['theor_mass'], charge_to_use)
                            if key in added:
                                continue
                            added.add(key)
                            if args.mva:
                                replicate_search_list[(d['id_scan']['rt'], ion_dict['ion'])].append((spectra_to_quant, d))
                            else:
                                ion_search_list.append((spectra_to_quant, d))
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
        mrm_added = set([])
        exclusion_masses = mrm_pair_info.loc[:,[i for i in mrm_pair_info.columns if i.lower() not in ('light', 'retention time')]].values.flatten() if args.mrm else set([])
        for scan_index, raw_scan_info in enumerate(raw_scans):
            target_scan = raw_scan_info['id_scan']
            quant_scan = raw_scan_info['quant_scan']
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
                lowest_label = min([j for i,v in mass_labels.items() for j in v])
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
            params = {'scan_info': raw_scan_info}
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

                    xic_peak_summary = OrderedDict()
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
                                # TODO: fix this terrible loop
                                for i, v in enumerate(RESULT_ORDER):
                                    for j in xic_peak_info:
                                        if '{}_{}'.format(label_name, j) == v[0]:
                                            try:
                                                xic_peak_summary[i].append(xic_peak_info[j])
                                            except KeyError:
                                                xic_peak_summary[i] = [xic_peak_info[j]]

                    if args.peaks_n == 1:
                        for index, values in six.iteritems(xic_peak_summary):
                            result_info = RESULT_ORDER[index]
                            if callable(result_info[-1]):
                                res_dict[result_info[0]] = result_info[-1](values)
                            else:
                                res_dict[result_info[0]] = values[0]
                res_dict['filename'] = filename
                # first entry of peak report is label, sort alphabetically
                peak_report.sort(key=operator.itemgetter(0))
                res_dict['peak_report'] = peak_report
                # This is the tsv output we provide
                res_list = [filename]+[res_dict.get(i[0], 'NA') for i in RESULT_ORDER]+['\t'.join(i) for i in peak_report]
                res = '{0}\n'.format('\t'.join(map(str, res_list)))
                out.write(res)
                out.flush()

                temp_file.write(pd.io.json.dumps({'res_dict': res_dict, 'html': result.get('html', {})}))
                temp_file.write('\n')
                temp_file.flush()

        if scans_to_export:
            raw_file = GuessIterator(filepath, full=True, store=False)
            for export_filename, scans in six.iteritems(export_mapping):
                with open(export_filename, 'w') as o:
                    raw_file.parser.writeScans(handle=o, scans=sorted(scans))

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
        result = pd.io.json.loads(j if isinstance(j, six.text_type) else j.decode('utf-8'))
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
    if calc_stats:
        perform_ml(data, mass_labels)
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
        peak_map = pd.io.json.dumps(peak_map)
        html_map = pd.io.json.dumps(html_map)
        html_out.write(html_template.safe_substitute({
            'peak_output': base64.b64encode(gzip.zlib.compress(peak_map if six.PY2 else six.binary_type(peak_map, 'utf-8'), 9)).decode('utf-8'),
            'html_output': base64.b64encode(gzip.zlib.compress(html_map if six.PY2 else six.binary_type(html_map, 'utf-8'), 9)).decode('utf-8'),
        }))

    os.remove(temp_file.name)
