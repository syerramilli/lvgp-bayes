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
- python >= 3.8
- torch >= 1.13.
- gpytorch >= 1.10.0
- numpy >= 1.21
- scipy >= 1.6
- jax >= 0.3.15
- numpyro >= 0.10

## Citing

```
@article{yerramilli2023fully,
  title={Fully Bayesian Inference for Latent Variable Gaussian Process Models},
  author={Yerramilli, Suraj and Iyer, Akshay and Chen, Wei and Apley, Daniel W},
  journal={SIAM/ASA Journal on Uncertainty Quantification},
  volume={11},
  number={4},
  pages={1357--1381},
  year={2023},
  publisher={SIAM}
}
```
