from __future__ import print_function
import os

# These exceptions are for building pyquant on lambdas, which parse
# the setup.py file. Sane builders will never hit this
include_dirs = []
try:
    import numpy

    include_dirs.append(numpy.get_include())
except ImportError:
    pass
from setuptools import (  # noqa: E402
    Extension,
    setup,
    find_packages,
)


# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))


extensions = [Extension(name="pyquant.cpeaks", sources=["pyquant/cpeaks.pyx"])]


setup(
    name="pyquant",
    version="0.4.3",
    packages=find_packages(),
    scripts=["scripts/pyQuant"],
    entry_points={"console_scripts": ["pyQuant = pyquant.command_line:run_pyquant",]},
    install_requires=[
        "cython ~= 0.29",
        "numpy ~= 1.21",
        "scipy ~= 1.7",
        "patsy ~= 0.5",
        "pythomics >= 0.3.46",
        "pandas ~= 1.3",
        "lxml ~= 4.7",
        "scikit-learn ~= 0.22",
        "simplejson ~= 3.17",
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
    include_dirs=include_dirs,
)
