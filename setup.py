import os
from setuptools import setup, find_packages

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setup(
    name='pyquant',
    version='0.1.0',
    packages=find_packages(),
    scripts=['scripts/pyQuant'],
    entry_points={'console_scripts': ['pyQuant = pyQuant.command_line:run_pyquant', ]},
    install_requires = ['pythomics', 'pandas', 'scipy', 'cython', 'lxml', 'scikit-learn', 'patsy'],
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
