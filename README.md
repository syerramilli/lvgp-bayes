# Latent variable Gaussian process models with fully Bayesian inference.

LVGP-Bayes is a Python library for estimating Latent variable Gaussian process (LVGP) models through fully Bayesian inference. This respository contains code to run the experiments in the paper [Fully Bayesian inference for latent variable Gaussian process models](https://arxiv.org/abs/2211.02218). 

For reproducing the experiments, refer to the each subdirectory in the `tests/` folder.

**Note:** The code is under an Academic and Non-Commerical Research use license.

## Installation

```
git clone https://github.com/syerramilli/lvgp-pytorch <path>
pip install <path>
```

*Note*: `<path>` is optional.

**Requirements**:
- python >= 3.7
- torch == 1.10.
- gpytorch == 1.6.0
- numpy >= 1.21
- scipy >= 1.6
- jax == 0.3.15
- numpyro == 0.10

## Citing

```
@article{yerramilli2022fully,
  title={Fully Bayesian inference for latent variable Gaussian process models},
  author={Yerramilli, Suraj and Iyer, Akshay and Chen, Wei and Apley, Daniel W},
  journal={arXiv preprint arXiv:2211.02218},
  year={2022}
}
```
