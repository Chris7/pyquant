import copy
import numpy as np
from datetime import datetime, timedelta
from multiprocessing import Process

from pythomics.proteomics.parsers import GuessIterator

from .logger import logger

class Reader(Process):
    def __init__(self, incoming, outgoing, raw_file=None, spline=None, rt_window=None, timeout_minutes=5):
        super(Reader, self).__init__()
        self.incoming = incoming
        self.outgoing = outgoing
        self.scan_dict = {}
        self.access_times = {}
        self.raw_path = raw_file
        self.spline = spline
        self.rt_window = rt_window
        self.timeout_minutes = timeout_minutes

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
                            scan_vals[:, 0] = scan_vals[:, 0] / (1 - self.spline(scan_vals[:, 0]) / 1e6)
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
                mz_end = out['vals'][-1, 0] + 1 if mz_end is None else mz_end
                out['vals'] = out['vals'][np.where((out['vals'][:, 0] >= mz_start) & (out['vals'][:, 0] <= mz_end))]
                self.outgoing[thread].put(out)
            else:
                self.outgoing[thread].put(d)
            now = datetime.now()
            self.access_times[scan_id] = now
            # evict scans we have not accessed in over 5 minutes
            cutoff = now - timedelta(minutes=self.timeout_minutes)
            to_delete = []
            for i, v in self.access_times.items():
                if v < cutoff:
                    del self.scan_dict[i]
                    to_delete.append(i)
            for i in sorted(to_delete, reverse=True):
                del self.access_times[i]
        logger.info('Reader done')