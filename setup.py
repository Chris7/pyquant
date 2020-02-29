from __future__ import print_function
import os

import numpy
from setuptools import (
    Extension,
    setup,
    find_packages,
)


# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))


extensions = [Extension(name="pyquant.cpeaks", sources=["pyquant/cpeaks.pyx"])]


setup(
    name="pyquant",
    version="0.4.0",
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
    setup_requires=["cython", "numpy"],
    ext_modules=extensions,
    include_dirs=[numpy.get_include()],
)
