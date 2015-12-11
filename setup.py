import os
from distutils.core import setup
from setuptools import find_packages

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setup(
    name='pyquant-ms',
    version='0.1.1',
    packages=find_packages(),
    scripts=['scripts/pyQuant'],
    entry_points={'console_scripts': ['pyQuant = pyquant.command_line:run_pyquant',]},
    install_requires = ['cython', 'numpy', 'scipy', 'patsy', 'pythomics', 'pandas', 'lxml', 'scikit-learn'],
    include_package_data=True,
    description='A framework for the analysis of quantitative mass spectrometry data',
    url='http://www.github.com/pandeylab/pyquant',
    author='Chris Mitchell <chris.mit7@gmail.com>',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
    ],
)
