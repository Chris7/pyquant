__author__ = 'chris'
import pkg_resources  # part of setuptools
import argparse
from pythomics.proteomics import config

version = pkg_resources.require('pyquant-ms')[0].version

description = """
This will quantify labeled peaks (such as SILAC) in ms1 spectra. It relies solely on the distance between peaks,
 which can correct for errors due to amino acid conversions.
"""

PEAK_RESOLUTION_RT_MODE = 'rt'
PEAK_RESOLUTION_COMMON_MODE = 'common-peak'

PEAK_FINDING_REL_MAX = 'relative-max'
PEAK_FINDING_DERIVATIVE = 'derivative'

PEAK_FIT_MODE_FAST = 'fast'
PEAK_FIT_MODE_AVERAGE = 'average'
PEAK_FIT_MODE_SLOW = 'slow'

pyquant_parser = argparse.ArgumentParser(prog='PyQuant v{}'.format(version), description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
pyquant_parser.add_argument('-p', help="Threads to run", type=int, default=1)
pyquant_parser.add_argument('--theo-xic', help=argparse.SUPPRESS, action='store_true')

raw_group = pyquant_parser.add_argument_group("Raw Data Parameters")
raw_group.add_argument('--scan-file', help="The scan file(s) for the raw data. If not provided, assumed to be in the directory of the processed/tabbed/peaklist file.", type=argparse.FileType('r'), nargs='*')
raw_group.add_argument('--scan-file-dir', help="The directory containing raw data.", type=str)
raw_group.add_argument('--precision', help="The precision for storing m/z values. Defaults to 6 decimal places.", type=int, default=6)
raw_group.add_argument('--precursor-ppm', help="The mass accuracy for the first monoisotopic peak in ppm.", type=float, default=5)
raw_group.add_argument('--isotope-ppm', help="The mass accuracy for the isotopic cluster.", type=float, default=2.5)
raw_group.add_argument('--spread', help="Assume there is spread of the isotopic label.", action='store_true')

search_group = pyquant_parser.add_argument_group("Search Information")
search_group.add_argument('--search-file', help='A search output or Proteome Discoverer msf file', type=argparse.FileType('rb'), required=False)
search_group.add_argument('--skip', help="If true, skip scans with missing files in the mapping.", action='store_true')
search_group.add_argument('--peptide', help="The peptide(s) to limit quantification to.", type=str, nargs='*')
search_group.add_argument('--peptide-file', help="A file of peptide(s) to limit quantification to.", type=argparse.FileType('r'))
search_group.add_argument('--scan', help="The scan(s) to limit quantification to.", type=str, nargs='*')

replicate_group = pyquant_parser.add_argument_group("Missing Value Analysis")
replicate_group.add_argument('--mva', help="Analyze files in 'missing value' mode.", action='store_true')
replicate_group.add_argument('--rt-window', help="The maximal deviation of a scan's retention time to be considered for analysis.", default=0.25, type=float)

label_group = pyquant_parser.add_argument_group("Labeling Information")
label_subgroup = label_group.add_mutually_exclusive_group()
label_subgroup.add_argument('--label-scheme', help='The file corresponding to the labeling scheme utilized.', type=argparse.FileType('r'))
label_subgroup.add_argument('--label-method', help='Predefined labeling schemes to use.', type=str, choices=sorted(config.LABEL_SCHEMES.keys()))
label_group.add_argument('--reference-label', help='The label to use as a reference (by default all comparisons are taken).', type=str)

tsv_group = pyquant_parser.add_argument_group('Tabbed File Input')
tsv_group.add_argument('--tsv', help='A delimited file containing scan information.', type=argparse.FileType('r'))
tsv_group.add_argument('--label', help='The column indicating the label state of the peptide. If not found, entry assumed to be light variant.', default='Labeling State')
tsv_group.add_argument('--peptide-col', help='The column indicating the peptide.', default='Peptide')
tsv_group.add_argument('--rt', help='The column indicating the retention time.', default='Retention time')
tsv_group.add_argument('--mz', help='The column indicating the MZ value of the precursor ion. This is not the MH+.', default='Light Precursor')
tsv_group.add_argument('--scan-col', help='The column indicating the scan corresponding to the ion.', default='MS2 Spectrum ID')
tsv_group.add_argument('--charge', help='The column indicating the charge state of the ion.', default='Charge')
tsv_group.add_argument('--source', help='The column indicating the raw file the scan is contained in.', default='Raw file')

ion_search_group = pyquant_parser.add_argument_group('Targetted Ion Search Parameters')
ion_search_group.add_argument('--msn-id', help='The ms level to search for the ion in. Default: 2 (ms2)', type=int, default=2)
ion_search_group.add_argument('--msn-quant-from', help='The ms level to quantify values from. i.e. if we are identifying an ion in ms2, we can quantify it in ms1 (or ms2). Default: msn value-1', type=int, default=None)
ion_search_group.add_argument('--msn-ion', help='M/Z values to search for in the scans. To search for multiple m/z values for a given ion, separate m/z values with a comma.', nargs='+', type=str)
ion_search_group.add_argument('--msn-ion-rt', help='RT values each ion is expected at.', nargs='+', type=float)
ion_search_group.add_argument('--msn-peaklist', help='A file containing peaks to search for in the scans.', type=argparse.FileType('rb'))
ion_search_group.add_argument('--msn-ppm', help='The error tolerance for identifying the ion(s).', type=float, default=200)
ion_search_group.add_argument('--msn-rt-window', help='The range of retention times for identifying the ion(s). (ex:  7.54-9.43)', type=str, nargs='+')
ion_search_group.add_argument('--msn-all-scans', help='Search for the ion across all scans (ie if you have 3 ions, you will have 3 results with one long XIC)', action='store_true')
ion_search_group.add_argument('--require-all-ions', help='If multiple ions are set (in the style of 93.15,105.15), all ions must be found in a scan.', action='store_true')

quant_parameters = pyquant_parser.add_argument_group('Quantification Parameters')
quant_parameters.add_argument('--quant-method', help='The process to use for quantification. Default: Integrate for ms1, sum for ms2+.', choices=['integrate', 'sum'], default=None)
quant_parameters.add_argument('--reporter-ion', help='Indicates that reporter ions are being used. As such, we only analyze a single scan.', action='store_true')
quant_parameters.add_argument('--isotopologue-limit', help='How many isotopologues to quantify', type=int, default=-1)
quant_parameters.add_argument('--overlapping-labels', help='This declares the mz values of labels will overlap. It is useful for data such as neucode, but not needed for only SILAC labeling.', action='store_true')
quant_parameters.add_argument('--labels-needed', help='How many labels need to be detected to quantify a scan (ie if you have a 2 state experiment and set this to 2, it will only quantify scans where both occur.', default=1, type=int)
quant_parameters.add_argument('--merge-labels', help='Merge labels together to a single XIC.', action='store_true')
quant_parameters.add_argument('--min-scans', help='How many quantification scans are needed to quantify a scan.', default=1, type=int)
quant_parameters.add_argument('--min-resolution', help='The minimal resolving power of a scan to consider for quantification. Useful for skipping low-res scans', default=0, type=float)
quant_parameters.add_argument('--no-mass-accuracy-correction', help='Disables the mass accuracy correction.', action='store_true')
quant_parameters.add_argument('--no-contaminant-detection', help='Disables routine to check if an ion is a contaminant of a nearby peptide (checks if its a likely isotopologue).', action='store_true')

peak_parameters = pyquant_parser.add_argument_group('Peak Fitting Parameters')
peak_parameters.add_argument('--peak-find-method', help='The method to use to identify peaks within data. For LC-MS, relative-max is usually best. For smooth data, derivative is better.', type=str, choices=(PEAK_FINDING_REL_MAX, PEAK_FINDING_DERIVATIVE), default=PEAK_FINDING_REL_MAX)
peak_parameters.add_argument(
    '--peak-find-mode',
    help='This picks some predefined parameters for various use cases. Fast is good for robust data with few peaks, slow is good for complex data with overlapping peaks of very different size.',
    type=str,
    choices=(PEAK_FIT_MODE_SLOW, PEAK_FIT_MODE_AVERAGE, PEAK_FIT_MODE_FAST),
    default=PEAK_FIT_MODE_AVERAGE
)
peak_parameters.add_argument('--gap-interpolation', help='This interpolates missing data in scans. The parameter should be a number that is the maximal gap size to fill (ie 2 means a gap of 2 scans). Can be useful for low intensity LC-MS data.', type=int, default=0)
peak_parameters.add_argument('--fit-baseline', help='Fit a separate line for the baseline of each peak.', action='store_true')
peak_parameters.add_argument('--peak-cutoff', help='The threshold from the initial retention time a peak can fall by before being discarded', type=float, default=0.05)
peak_parameters.add_argument('--max-peaks', help='The maximal number of peaks to detect per scan. A lower value can help with very noisy data.', type=int, default=-1)
peak_parameters.add_argument('--peaks-n', help='The number of peaks to report per scan. Useful for ions with multiple elution times.', type=int, default=1)
peak_parameters.add_argument('--no-rt-guide', help='Do not use the retention time to bias for peaks containing the MS trigger time.', action='store_true')
peak_parameters.add_argument('--snr-filter', help='Filter peaks below a given SNR.', type=float, default=0)
peak_parameters.add_argument('--zscore-filter', help='Peaks below a given z-score are excluded.', type=float, default=0)
peak_parameters.add_argument('--filter-width', help='The window size for snr/zscore filtering. Default: entire scan', type=float, default=0)
peak_parameters.add_argument('--r2-cutoff', help='The minimal R^2 for a peak to be kept. Should be a value between 0 and 1', type=float, default=None)
peak_parameters.add_argument('--intensity-filter', help='Filter peaks whose peak are below a given intensity.', type=float, default=0)
peak_parameters.add_argument('--percentile-filter', help='Filter peaks whose peak are below a given percentile of the data.', type=float, default=0)
peak_parameters.add_argument('--min-peak-separation', help='Peaks separated by less than this distance will be combined. For very crisp data, set this to a lower number. (minimal value is 1)', type=int, default=5)
peak_parameters.add_argument('--disable-peak-filtering', help='This will disable smoothing of data prior to peak finding. If you have very good LC, this may be used to identify small peaks.', action='store_true')
peak_parameters.add_argument('--merge-isotopes', help='Merge Isotopologues together prior to fitting.', action='store_true')
peak_parameters.add_argument('--peak-resolution-mode', help='The method to use to resolve peaks across multiple XICs', choices=(PEAK_RESOLUTION_RT_MODE, PEAK_RESOLUTION_COMMON_MODE), type=str, default='common-peak')

# Deprecated parameters
peak_parameters.add_argument('--remove-baseline', help=argparse.SUPPRESS, action='store_true', dest='fit_baseline')


xic_parameters = pyquant_parser.add_argument_group('XIC Options')
xic_parameters.add_argument('--xic-snr', help='When the SNR of the XIC falls below this, stop searching for more data. Useful for escaping from noisy shoulders and contaminants.', type=float, default=1.0)
xic_parameters.add_argument('--xic-missing-ion-count', help='This specifies how many consequtive scans an ion can be missing for until it is no longer considered.', type=int, default=1)
xic_parameters.add_argument('--xic-window-size', help='When the number of scans in a given direction from the initial datapoint of an XIC passes this, stop. Default is -1 (disabled). Useful for removing contaminants', type=int, default=-1)
xic_parameters.add_argument('--xic-smooth', help='Prior to fitting, smooth data with a Gaussian filter.', action='store_true')
xic_parameters.add_argument('--export-msn', help='This will export spectra of a given MSN that were used to provide the quantification.', action='store_false')


mrm_parameters = pyquant_parser.add_argument_group('SRM/MRM Parameters')
#'A file indicating light and heavy peptide pairs, and optionally the known elution time.'
mrm_parameters.add_argument('--mrm-map', help=argparse.SUPPRESS, type=argparse.FileType('r'))

output_group = pyquant_parser.add_argument_group("Output Options")
output_group.add_argument('--debug', help="This will output debug information.", action='store_true')
output_group.add_argument('--html', help="Output a HTML table summary.", action='store_true')
output_group.add_argument('--resume', help="Will resume from the last run. Only works if not directing output to stdout.", action='store_true')
output_group.add_argument('--sample', help="How much of the data to sample. Enter as a decimal (ie 1.0 for everything, 0.1 for 10%%)", type=float, default=1.0)
output_group.add_argument('--disable-stats', help="Disable confidence statistics on data.", action='store_true')
output_group.add_argument('--no-ratios', help="Disable reporting of ratios in output.", action='store_true')
output_group.add_argument('-o', '--out', nargs='?', help='The prefix for the file output', type=str)

PER_PEAK = 'per-peak'
PER_FILE = 'per-file'
PER_ID = 'per-id'

spectra_output = pyquant_parser.add_argument_group("Spectra Output Options")
spectra_output.add_argument('--export-mzml', help='Create an mzml file of spectra contained within each peak.', action='store_true')
spectra_output.add_argument('--export-mode', help='How to export the scans. per-peak: A mzML per peak identified. per-id: A mzML per ion identified (each row of the output gets an mzML). per-file: All scans matched per raw file.', type=str, default='per-peak', choices={PER_PEAK, PER_ID, PER_FILE})

convenience_group = pyquant_parser.add_argument_group('Convenience Parameters')
convenience_group.add_argument('--neucode', help='This will select parameters specific for neucode. Note: You still must define a labeling scheme.', action='store_true')
convenience_group.add_argument('--isobaric-tags', help='This will select parameters specific for isobaric tag based labeling (TMT/iTRAQ).', action='store_true')
convenience_group.add_argument('--ms3', help='This will select parameters specific for ms3 based quantification.', action='store_true')
convenience_group.add_argument('--maxquant', help='This will select parameters specific for a MaxQuant evidence file.', action='store_true')
convenience_group.add_argument('--gcms', help='This will select parameters specific for ion identification and quantification in GCMS experiments.', action='store_true')
#'This will select parameters specific for Selective/Multiple Reaction Monitoring (SRM/MRM).'
convenience_group.add_argument('--mrm', help=argparse.SUPPRESS, action='store_true')

