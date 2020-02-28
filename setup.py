from __future__ import print_function

import os
from distutils.core import setup
from setuptools import find_packages


try:
    import numpy

    NUMPY = True
except ImportError:
    NUMPY = False
    print("NUMPY NOT INSTALLED")

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))


class defer_cythonize(list):
    def __init__(self, callback):
        self._list, self.callback = None, callback

    def c_list(self):
        if self._list is None:
            self._list = self.callback()
        return self._list

    def __iter__(self):
        for elem in self.c_list():
            yield elem

    def __getitem__(self, ii):
        return self.c_list()[ii]

    def __len__(self):
        return len(self.c_list())


def extensions():
    from Cython.Build import cythonize

    return cythonize("pyquant/*.pyx")


setup(
    name="pyquant",
    version="0.3.2",
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
    ext_modules=defer_cythonize(extensions),
    include_dirs=[numpy.get_include()] if NUMPY else [],
)
