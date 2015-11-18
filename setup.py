import os
from setuptools import setup, find_packages
try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
    CYTHON=True
except ImportError:
    CYTHON=False

try:
    import numpy
    np_install = True
except ImportError:
    np_install = False

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setup(
    name='pyquant',
    version='0.1.0',
    packages=find_packages(),
    scripts=['scripts/pyQuant'],
    entry_points={'console_scripts': ['pyQuant = pyquant.command_line:run_pyquant',]},
    install_requires = ['pythomics', 'pandas', 'scipy', 'cython', 'lxml', 'scikit-learn', 'patsy', 'numpy'],
    include_package_data=True,
    description='A framework for the analysis of quantitative mass spectrometry data',
    url='http://www.github.com/pandeylab/pyquant',
    author='Chris Mitchell <chris.mit7@gmail.com>',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
    ],
    ext_modules = cythonize("pyquant/*.pyx") if CYTHON else None,
    include_dirs=[numpy.get_include()] if np_install else None,
)
