# LVGP and other non-standard domain GPs in PyTorch

`lvgp-pytorch` is an implementation of Latent variable Gaussian process and other non-standard domain Gaussian processes in PyTorch. It is built on top of [GPyTorch](https://github.com/cornellius-gp/gpytorch). 

## Installation

```
git clone https://github.com/syerramilli/lvgp-pytorch <path>
pip install <path>
```

*Note*: `<path>` is optional.

**Requirements**:
- python >= 3.7
- torch == 1.7
- gpytorch == 1.2.1
- numpy == 1.20
- scipy == 1.6 (for `scipy.optimize`)

*Note*: Make sure to install the `1.2.1` version of gpytorch. There are some incompatibilities with version `1.3.1` version.

## Major model classes

- Exact GP models (regression)
    - `GPR`: the base GP regression model
    - `LVGPR`: the latent variable GP regression model for domains with qualitative variables

GPU functionality is yet to be tested. However, we expect no hassles with training the models on GPUs.

## Examples

<TODO>
