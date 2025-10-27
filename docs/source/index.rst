.. LVGP-Bayes documentation master file

LVGP-Bayes Documentation
========================

LVGP-Bayes is a Python library for estimating Latent Variable Gaussian Process (LVGP) models through fully Bayesian inference.

This package implements methods from the paper `Fully Bayesian Inference for Latent Variable Gaussian Process Models <https://arxiv.org/abs/2211.02218>`_ published in SIAM/ASA Journal on Uncertainty Quantification (2023).

🚀 Key Features
---------------

- **Latent Variable Gaussian Processes**: Handle mixed categorical and quantitative inputs
- **Fully Bayesian Inference**: MCMC sampling using NumPyro with HMC/NUTS
- **Maximum A Posteriori (MAP) Estimation**: Fast optimization using SciPy
- **Sparse Approximations**: FITC and VFE methods for scalability
- **Custom Kernels and Priors**: Specialized implementations for Bayesian GP models

.. warning::
   **License**: This software is for **Academic and Non-Commercial Research Use Only**.

📖 Quick Start
---------------

Installation
^^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/syerramilli/lvgp-bayes
   cd lvgp-bayes
   pip install -e .

For running experiments:

.. code-block:: bash

   pip install -e ".[experiments]"

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   import torch
   from lvgp_bayes.models import LVGPR
   from lvgp_bayes.optim import fit_model_scipy, run_hmc_numpyro

   # Create an LVGP model
   model = LVGPR(
       train_x=train_x,
       train_y=train_y,
       qual_index=[0, 1],      # indices of categorical variables
       quant_index=[2, 3, 4],  # indices of quantitative variables
       num_levels_per_var=[5, 3],  # number of levels for each categorical
       lv_dim=2  # latent space dimension
   )

   # MAP estimation
   model = fit_model_scipy(model, max_iter=1000)

   # Or fully Bayesian inference
   samples = run_hmc_numpyro(
       model,
       num_samples=1000,
       warmup_steps=1000
   )


📚 Documentation Contents
-------------------------

.. toctree::
   :maxdepth: 2
   :caption: Examples:

   examples/index

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   models
   optim
   utils

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
