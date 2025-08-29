"""
Setup script for the FlowUtils package - Pure Python implementation
"""
from setuptools import setup

# read in version string
VERSION_FILE = 'flowutils/_version.py'
__version__ = None  # to avoid inspection warning and check if __version__ was loaded
exec(open(VERSION_FILE).read())

if __version__ is None:
    raise RuntimeError("__version__ string not found in file %s" % VERSION_FILE)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='FlowUtils',
    version=__version__,
    packages=['flowutils'],
    package_data={'': []},
    description='Pure Python Flow Cytometry Transforms (Logicle & Hyperlog)',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Scott White',
    author_email='whitews@gmail.com',
    license='BSD',
    url="https://github.com/whitews/flowutils",
    install_requires=['numpy>=1.22,<2.0', 'scipy>=1.7'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8', 
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Bio-Informatics'
    ],
    python_requires='>=3.8',
)
