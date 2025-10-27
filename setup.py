# Initialize setup
import os
import sys
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

# Read requirements
with open('requirements.txt') as fh:
    requirements = [line.strip() for line in fh.readlines() if line.strip() and not line.startswith('#')]

with open('exp_requirements.txt') as f:
    exp_requirements = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]

with open('docs/requirements.txt') as f:
    docs_requirements = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]

# Read long description from README
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='lvgp-bayes',
    version='0.3.0',
    description='Latent Variable Gaussian Process models with fully Bayesian inference',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/syerramilli/lvgp-bayes',
    author='Suraj Yerramilli',
    author_email='surajyerramilli@gmail.com',
    license='Proprietary - Academic and Non-Commercial Research Use Only',

    # Python version requirement
    python_requires='>=3.10',

    # Package discovery
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests", "experiments", "notebooks"]),

    # Dependencies
    install_requires=requirements,
    extras_require={
        'experiments': exp_requirements,
        'docs': docs_requirements,
    },

    # Classifiers
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: Other/Proprietary License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],

    # Keywords
    keywords='gaussian-processes bayesian-inference latent-variables machine-learning',

    zip_safe=False
)