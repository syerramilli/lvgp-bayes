.. _api_optim:

Optimization and Inference
===========================

Methods for parameter estimation via MAP estimation and fully Bayesian MCMC inference.


MAP Estimation with SciPy
--------------------------

.. autofunction:: lvgp_bayes.optim.fit_model_scipy

.. autofunction:: lvgp_bayes.optim.mll_scipy.marginal_log_likelihood


Fully Bayesian Inference with NumPyro
--------------------------------------

.. autofunction:: lvgp_bayes.optim.run_hmc_numpyro
