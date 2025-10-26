# Initialize setup
import os
import sys
from setuptools import setup,find_packages
here = os.path.abspath(os.path.dirname(__file__))

with open('requirements.txt') as fh:
    requirements = [line.strip() for line in fh.readlines()]

with open('exp_requirements.txt') as f:
      exp_requirements = f.read().splitlines()

setup(name='lvgp-bayes',
      version='0.3.0',
      description='LVGP models with bayesian inference',
      url='http://github.com/syerramilli/lvgp-bayes',
      author='Suraj Yerramilli',
      author_email='surajyerramilli@gmail.com',
      license='MIT',
      packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests", "experiments/*", "notebooks"]),
      install_requires=requirements,
      extras_require={
          'experiments': exp_requirements
      },
      zip_safe=False)