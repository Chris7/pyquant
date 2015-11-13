import sys
from distutils.version import StrictVersion

PY_FULL_VERSION = StrictVersion('{}.{}.{}'.format(sys.version_info.major, sys.version_info.minor, sys.version_info.micro))
PY_MINOR_VERSION = StrictVersion('{}.{}'.format(sys.version_info.major, sys.version_info.minor))
PY3 = StrictVersion('3.0.0')
PY2 = StrictVersion('2.7.0')

if PY_FULL_VERSION >= PY3:
    import queue as Queue
elif PY_FULL_VERSION >= PY2:
    import Queue