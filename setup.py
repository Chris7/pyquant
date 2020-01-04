from __future__ import print_function

import os
from distutils.core import setup
from setuptools import find_packages

try:
    from Cython.Build import cythonize
    import Cython.Distutils

    CYTHON = True
except ImportError:
    CYTHON = False
    print("CYTHON UNAVAILABLE")
try:
    import numpy

    NUMPY = True
except ImportError:
    NUMPY = False
    print("NUMPY NOT INSTALLED")

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setup(
    name="pyquant",
    version="0.3.0rc3",
    packages=find_packages(),
    scripts=["scripts/pyQuant"],
    entry_points={"console_scripts": ["pyQuant = pyquant.command_line:run_pyquant",]},
    install_requires=[
        "cython",
        "numpy",
        "scipy >= 0.18.*",
        "patsy",
        "pythomics >= 0.3.41",
        "pandas >= 0.24.0",
        "lxml",
        "scikit-learn",
        "simplejson",
    ],
    include_package_data=True,
    description="A framework for the analysis of quantitative mass spectrometry data",
    url="http://www.github.com/chris7/pyquant",
    author="Chris Mitchell <chris.mit7@gmail.com>",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    setup_requires=["cython",],
    cmdclass={"build_ext": Cython.Distutils.build_ext} if CYTHON else {},
    ext_modules=cythonize("pyquant/*.pyx") if CYTHON else [],
    include_dirs=[numpy.get_include()] if NUMPY else [],
)
