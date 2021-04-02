#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages

setup(
    name='se3_distributions',
    version='0.1dev',
    author='Brian Okorn',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    description='Estimate pose distributions based on latent features',
    long_description=open('README.md').read(),
    package_data = {'': ['*.mat', '*.blend','*.npy']},
)


### ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD
#
#from distutils.core import setup
#from catkin_pkg.python_setup import generate_distutils_setup
#
## fetch values from package.xml
#setup_args = generate_distutils_setup(
#    packages=['generic_pose'],
#    package_dir={'': 'src'},
#)
#
#setup(**setup_args)
##
