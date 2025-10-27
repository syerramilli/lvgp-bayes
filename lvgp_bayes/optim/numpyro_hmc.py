import warnings
import torch
import numpy as np
import math
import jax
from jax import vmap
import jax.numpy as jnp
import jax.random as random
from jax.scipy import linalg

import numpyro
import numpyro.distributions as dist
from numpyro.infer import (
    MCMC,
    NUTS,
    init_to_sample,
    init_to_value
)

from .numpryo_dists import MollifiedUniform

from lvgp_bayes.models import GPR,LVGPR
from lvgp_bayes.models.sparselvgp import SparseLVGPR,theta_to_weights

from typing import Optional, Dict, Tuple
from collections import OrderedDict
from copy import deepcopy

def cov_map(cov_func, xs, xs2=None):
    """Compute covariance matrix using JAX vmap for efficient vectorization.

    Applies a pairwise covariance function to compute the full covariance matrix
    between data points. Uses JAX's vmap for automatic vectorization.

    Args:
        cov_func (callable): Covariance function that takes two data points and
            returns a scalar covariance value.
        xs (jax.numpy.ndarray): Array of data points with shape (n, d) where n is
            the number of points and d is the dimensionality.
        xs2 (jax.numpy.ndarray, optional): Second array of data points with shape
            (m, d). If provided, computes cross-covariance matrix K(xs, xs2).
            If None, computes K(xs, xs) (default).

    Returns:
        jax.numpy.ndarray: Covariance matrix of shape (n, n) if xs2 is None,
            or (n, m) if xs2 is provided, where element [i, j] equals
            cov_func(xs[i], xs2[j]).

    Note:
        This is an internal helper function used by NumPyro model definitions.
    """
    if xs2 is None:
        return vmap(lambda x: vmap(lambda y: cov_func(x, y))(xs))(xs)
    else:
        return vmap(lambda x: vmap(lambda y: cov_func(x, y))(xs))(xs2).T
    
def rbfkernel(x1, x2):
    """Radial Basis Function (RBF) / Squared Exponential kernel.

    Computes the unnormalized RBF kernel between two input vectors.
    The kernel is stationary and depends only on the Euclidean distance
    between inputs: k(x1, x2) = exp(-0.5 * ||x1 - x2||^2).

    Args:
        x1 (jax.numpy.ndarray): First input vector of shape (d,).
        x2 (jax.numpy.ndarray): Second input vector of shape (d,).

    Returns:
        float: Scalar kernel value between 0 and 1.

    Note:
        This kernel assumes inputs are already scaled by lengthscale.
        For use with NumPyro models only.
    """
    return jnp.exp(-0.5*jnp.sum((x1 - x2)**2))

def matern52kernel(x1, x2):
    """Matérn kernel with smoothness parameter ν = 5/2.

    Computes the Matérn 5/2 kernel, which is twice differentiable.
    The kernel has the form: k(r) = (1 + √5·r + 5·r²/3) · exp(-√5·r),
    where r is the Euclidean distance between inputs.

    Args:
        x1 (jax.numpy.ndarray): First input vector of shape (d,).
        x2 (jax.numpy.ndarray): Second input vector of shape (d,).

    Returns:
        float: Scalar kernel value. Approaches 1 as r → 0 and decays to 0 as r → ∞.

    Note:
        A small jitter (1e-12) is added for numerical stability when computing distance.
        This kernel assumes inputs are already scaled by lengthscale.
    """
    r = jnp.sqrt(jnp.sum((x1 - x2)**2) + 1e-12)
    exp_component = jnp.exp(-math.sqrt(5)*r)
    constant_component =1 + math.sqrt(5)*r + 5/3*(r**2)
    return constant_component*exp_component

# define global dictionary of kernels
kernel_names = {
    'rbfkernel':rbfkernel,
    #'matern32kernel':matern32kernel,
    'matern52kernel':matern52kernel
}

def translate(Z):
    """Translate latent variable embeddings to remove location invariance.

    Centers the latent embeddings by subtracting the first embedding point,
    ensuring the first point is at the origin.

    Args:
        Z (jax.numpy.ndarray): Latent embeddings with shape (n_levels, lv_dim).

    Returns:
        jax.numpy.ndarray: Translated embeddings with shape (n_levels, lv_dim)
            where the first point is at the origin.

    Note:
        This is part of the identifiability constraints for latent variable mappings.
    """
    return Z-Z[...,[0],:]

def translate_and_rotate(Z):
    """Apply translation and rotation to remove invariances in latent embeddings.

    Removes both location and rotation invariance from 2D latent variable embeddings
    by: (1) translating to place the first point at origin, and (2) rotating to
    align the second point with a canonical orientation.

    Args:
        Z (jax.numpy.ndarray): Raw latent embeddings with shape (n_levels, 2).
            Must be 2-dimensional (lv_dim=2).

    Returns:
        jax.numpy.ndarray: Transformed embeddings with shape (n_levels, 2) where:
            - First point is at the origin
            - Second point lies on a canonical axis (removes rotation)

    Note:
        This transformation ensures identifiability of the latent variable mapping
        by removing degeneracies due to translation and rotation. Used in NumPyro
        models for LVGP inference. Only works for 2D latent spaces.
    """
    Zcen = Z-Z[...,[0],:]
    theta = jnp.arctan(Zcen[...,[1],[1]]/Zcen[...,[1],[0]])
    c,s = jnp.cos(theta),jnp.sin(theta)
    R = jnp.vstack([
        jnp.column_stack([c,-s]),
        jnp.column_stack([s,c])
    ])
    return Zcen @ R

class ExpHalfCauchy(dist.TransformedDistribution):
    """Exponentially transformed Half-Cauchy distribution.

    Creates a distribution where X = log(Y) and Y ~ HalfCauchy(scale).
    This is equivalent to applying the inverse exponential transform to a
    Half-Cauchy distribution, useful for priors on log-scale parameters.

    Args:
        scale (float): Scale parameter for the base Half-Cauchy distribution.

    Note:
        Used as a prior for noise parameters in NumPyro GP models. The log
        transformation ensures the parameter stays positive while allowing
        wide tails for robust inference.
    """
    def __init__(self, scale):
        base_dist = dist.HalfCauchy(scale)
        super().__init__(
            base_dist, dist.transforms.ExpTransform().inv
        )

def get_samples(samples, num_samples=None, group_by_chain=False):
    """Thin MCMC samples to reduce storage requirements.

    Selects a subset of MCMC samples via systematic thinning (equally spaced indices).
    This reduces autocorrelation and storage requirements while maintaining
    representative posterior samples.

    Args:
        samples (dict): Dictionary of MCMC samples where keys are parameter names
            and values are PyTorch tensors of shape (n_samples, ...).
        num_samples (int, optional): Number of samples to retain after thinning.
            If None, all samples are returned. Defaults to None.
        group_by_chain (bool, optional): Whether to preserve chain dimension.
            Currently unused but maintained for API compatibility. Defaults to False.

    Returns:
        dict: Dictionary of thinned samples with the same keys as input.
            Each tensor has shape (num_samples, ...) if num_samples is specified,
            otherwise unchanged.

    Note:
        Samples are selected in reverse order (most recent first) using linearly
        spaced indices across the full sample range. This is an internal helper
        for post-processing MCMC output.
    """
    if num_samples is not None:
        batch_dim = 0
        sample_tensor = list(samples.values())[0]
        batch_size, device = sample_tensor.shape[batch_dim], sample_tensor.device
        idxs = torch.linspace(0, batch_size-1, num_samples, dtype=torch.long, device=device).flip(0)
        samples = {k: v.index_select(batch_dim, idxs) for k, v in samples.items()}
    return samples

def run_hmc_numpyro(
    model,
    num_samples: int = 500,
    warmup_steps: int = 500,
    num_model_samples: int = 100,
    disable_progbar: bool = True,
    num_chains: int = 1,
    max_tree_depth: int = 5,
    initialize_from_state: bool = False,
    seed: int = 0
):
    """Run fully Bayesian MCMC inference for Gaussian Process models using NumPyro.

    Performs Hamiltonian Monte Carlo (HMC) inference using the No-U-Turn Sampler (NUTS)
    to obtain posterior samples for GP hyperparameters. Automatically detects model type
    and uses the appropriate NumPyro probabilistic program.

    Supports three model types:
        - Standard GP (GPR): Basic Gaussian Process regression
        - Latent Variable GP (LVGPR): GP with categorical inputs via latent mappings
        - Sparse LVGP (SparseLVGPR): LVGP with FITC/VFE approximations for scalability

    Args:
        model (GPR, LVGPR, or SparseLVGPR): A fitted GP model instance. The model's
            current parameter values are used for initialization if
            ``initialize_from_state=True``. Model type is auto-detected.
        num_samples (int, optional): Number of MCMC samples to collect after warmup.
            More samples improve posterior approximation but increase runtime.
            Defaults to 500.
        warmup_steps (int, optional): Number of warmup/adaptation steps for NUTS.
            During warmup, step size and mass matrix are tuned. Defaults to 500.
        num_model_samples (int, optional): Number of posterior samples to retain after
            thinning. The final model will use this many posterior samples for
            predictions. Should be less than ``num_samples``. Defaults to 100.
        disable_progbar (bool, optional): If True, disables the progress bar during
            sampling. Useful for cleaner output in scripts. Defaults to True.
        num_chains (int, optional): Number of independent MCMC chains to run. Multiple
            chains help diagnose convergence. Currently runs sequentially. Defaults to 1.
        max_tree_depth (int, optional): Maximum tree depth for NUTS. Larger values allow
            longer trajectories but increase computation. Defaults to 5.
        initialize_from_state (bool, optional): If True, initializes MCMC from the
            current model parameters. If False, samples initial values from priors.
            Defaults to False.
        seed (int, optional): Random seed for reproducibility. Defaults to 0.

    Returns:
        numpyro.infer.MCMC: MCMC run object containing the samples and diagnostics.
            Use ``mcmc.get_samples()`` to access posterior samples, or
            ``mcmc.print_summary()`` to see convergence diagnostics.

    Note:
        - It is recommend to configure JAX for 64-bit precision before calling this function:
          ``jax.config.update("jax_enable_x64", True)``
        - The input model is modified in-place with posterior samples loaded into
          its state dict. After this call, the model can be used for Bayesian
          predictions with uncertainty quantification.
        - For convergence diagnostics, use ``numpyro.diagnostics.summary(mcmc.get_samples())``.

    Raises:
        ValueError: If model type is not supported (must be GPR, LVGPR, or SparseLVGPR).
    """
    kwargs = {
        'x':jnp.array(model.train_inputs[0].numpy()),
        'y':jnp.array(model.train_targets.numpy()),
        
    }
    dense_mass=False

    if isinstance(model, SparseLVGPR):
        numpyro_model = numpyro_fitc_lvgp
        with torch.no_grad():
            num_levels_per_var = [layer.raw_latents.shape[0] for layer in model.lv_mapping_layers]
            quant_inducing = torch.sigmoid(model.raw_quant_inducing).clone().numpy()
            qual_weights = [
                theta_to_weights(getattr(model,'raw_thetas%d'%k)).numpy() for k in range(len(model.qual_index))
            ]
        
        kwargs.update({
            'qual_index':model.qual_index.tolist(),
            'quant_index':model.quant_index.tolist(),
            'num_levels_per_var':num_levels_per_var,
            'quant_inducing':quant_inducing,
            'qual_weights':qual_weights,
            'approx':model.approx,
            'jitter':model.raw_noise_constraint.lower_bound.item()
        })  

    elif isinstance(model, LVGPR):
        numpyro_model = numpyro_lvgp
        with torch.no_grad():
            num_levels_per_var = [layer.raw_latents.shape[0] for layer in model.lv_mapping_layers]
        
        # alpha and beta aren't expected to change
        # this is only for testing different values of alpha and beta
        alpha = model.lv_mapping_layers[0].raw_precision_prior.base_dist.concentration.item()
        beta = model.lv_mapping_layers[0].raw_precision_prior.base_dist.rate.item()

        # additional kwargs
        kwargs.update({
            'qual_index':model.qual_index.tolist(),
            'quant_index':model.quant_index.tolist(),
            'num_levels_per_var':num_levels_per_var,
            'jitter':model.likelihood.noise_covar.raw_noise_constraint.lower_bound.item(),
            'alpha':alpha,
            'beta':beta
        })

    else:
        numpyro_model = numpyro_gpr

        kwargs.update({
            'jitter':model.likelihood.noise_covar.raw_noise_constraint.lower_bound.item(),
            'kernel':model.covar_module.base_kernel.__class__.__name__.lower()
        })

    if initialize_from_state:
        init_values = {}
        for name,module,_,closure,_ in model.named_priors():
            init_values[name[:-6]] = jnp.array(closure(module).detach().clone().numpy())
        init_strategy = init_to_value(values=init_values)
    else:
        init_strategy = init_to_sample
    
    kernel = NUTS(
        numpyro_model,
        step_size=0.1,
        adapt_step_size=True,
        init_strategy=init_strategy,
        max_tree_depth=max_tree_depth,
        dense_mass=dense_mass
    )
    mcmc_runs = MCMC(
        kernel,
        num_warmup=warmup_steps,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar= (not disable_progbar),
        chain_method='sequential',
        #jit_model_args=True
    )
    mcmc_runs.run(random.PRNGKey(seed),**kwargs)
    samples = {
        k:torch.from_numpy(np.array(v)).to(model.train_targets) for k,v in mcmc_runs.get_samples().items()
    }
    samples = {k:v for k,v in get_samples(samples,num_model_samples).items()}

    model.train_inputs = tuple(tri.unsqueeze(0).expand(num_model_samples, *tri.shape) for tri in model.train_inputs)
    model.train_targets = model.train_targets.unsqueeze(0).expand(num_model_samples, *model.train_targets.shape)

    state_dict = deepcopy(model.state_dict())
    state_dict.update(samples)

    if isinstance(model, SparseLVGPR):
        for k in range(len(model.qual_index)):
            state_dict['raw_thetas%d'%k] = (
                state_dict['raw_thetas%d'%k].clone()
                .unsqueeze(0).repeat(num_model_samples,1,1)
            )
        
        state_dict['raw_quant_inducing'] = (
            state_dict['raw_quant_inducing']
            .unsqueeze(0).repeat(num_model_samples,1,1)
        )

    # Load parameters without standard shape checking.
    model.load_strict_shapes(False)
    model.load_state_dict(state_dict)

    return mcmc_runs

########
# NumPyro Probabilistic Models
########
# These functions define probabilistic programs for NumPyro's MCMC inference.
# They specify the prior distributions and likelihood for different GP variants.

def numpyro_gpr(
    x, y, kernel='rbfkernel', jitter=1e-6
):
    """NumPyro probabilistic program for standard Gaussian Process regression.

    Defines the Bayesian model for GP regression with RBF or Matérn kernel.
    Specifies priors on hyperparameters and the GP likelihood.

    Args:
        x (jax.numpy.ndarray): Training inputs with shape (n, d).
        y (jax.numpy.ndarray): Training targets with shape (n,).
        kernel (str, optional): Kernel type, either 'rbfkernel' or 'matern52kernel'.
            Defaults to 'rbfkernel'.
        jitter (float, optional): Small positive value added to diagonal for numerical
            stability. Defaults to 1e-6.

    Note:
        This is an internal function used by :func:`run_hmc_numpyro`.
        Do not call directly. Prior specifications:
            - mean: Normal(0, 1)
            - outputscale (log): Normal(0, 1)
            - noise (log): ExpHalfCauchy(1e-2)
            - lengthscale (log): MollifiedUniform(log(0.1), log(10))
    """
    mean = numpyro.sample('mean_module.raw_constant',dist.Normal(0,1))
    outputscale = numpyro.sample("covar_module.raw_outputscale", dist.Normal(0.0, 1))
    noise = numpyro.sample(
        "likelihood.noise_covar.raw_noise",
        ExpHalfCauchy(1e-2).expand([1])
    )
    
    num_inputs = x.shape[1]
    lengthscale = numpyro.sample(
        'covar_module.base_kernel.raw_lengthscale',
        MollifiedUniform(math.log(0.1),math.log(10)).expand([1,len(quant_index)])
    )

    x2 = x/jnp.exp(lengthscale)

    # compute kernel
    k = jnp.exp(outputscale)*cov_map(kernel_names[kernel],x2)
    # add noise and jitter
    k += (jnp.exp(noise)+jitter)*jnp.eye(x.shape[0])
    
    # sample Y according to the standard gaussian process formula
    numpyro.sample(
        "y",
        dist.MultivariateNormal(loc=mean*jnp.ones(x.shape[0]), covariance_matrix=k),
        obs=y,
    )

def numpyro_lvgp(
    x, y, qual_index, quant_index, num_levels_per_var,
    jitter=1e-6, alpha=2., beta=1.
):
    """NumPyro probabilistic program for Latent Variable Gaussian Process.

    Defines the Bayesian model for LVGP with categorical inputs mapped to latent
    space. Includes priors on latent embeddings, precision parameters, and GP
    hyperparameters.

    Args:
        x (jax.numpy.ndarray): Training inputs with shape (n, d) where first columns
            are categorical (qual_index) and remaining are quantitative (quant_index).
        y (jax.numpy.ndarray): Training targets with shape (n,).
        qual_index (list of int): Column indices for categorical variables.
        quant_index (list of int): Column indices for quantitative variables.
        num_levels_per_var (list of int): Number of levels for each categorical variable.
        jitter (float, optional): Numerical stability constant. Defaults to 1e-6.
        alpha (float, optional): Shape parameter for Gamma prior on precision.
            Defaults to 2.0.
        beta (float, optional): Rate parameter for Gamma prior on precision.
            Defaults to 1.0.

    Note:
        This is an internal function used by :func:`run_hmc_numpyro`.
        Do not call directly. Latent embeddings are constrained via
        :func:`translate_and_rotate` to remove invariances. Prior specifications:
            - Latent embeddings: Normal(0, 1) with precision scaling
            - Precision (log): InverseGamma(alpha, beta) via log transform
            - GP hyperparameters: Same as :func:`numpyro_gpr`
    """
    mean = numpyro.sample('mean_module.raw_constant',dist.Normal(0,1))
    outputscale = numpyro.sample("covar_module.raw_outputscale", dist.Normal(0.0, 1))
    noise = numpyro.sample(
        "likelihood.noise_covar.raw_noise",
        ExpHalfCauchy(1e-2).expand([1])
    )

    num_qual = len(qual_index)

    precisions = [
        numpyro.sample(
            'lv_mapping_layers.%d.raw_precision'%i,
            dist.TransformedDistribution(
                dist.Gamma(alpha,beta), 
                dist.transforms.ExpTransform().inv
            ).expand([1])
        ) for i in range(num_qual)
    ]

    raw_latents =[
        numpyro.sample(
            'lv_mapping_layers.%d.raw_latents'%i,
            dist.Normal(0,1).expand([num_levels_per_var[i],2])
        ) for i in range(num_qual)
    ]

    
    latents = ([
        translate_and_rotate(
            raw_latents[i]/jnp.sqrt(jnp.exp(precisions[i]))/math.sqrt(num_levels_per_var[i])
        ) for i in range(num_qual)
    ])
    
    x2 = jnp.column_stack([
        jnp.take(
            latents[i],x[:,qual_index[i]].astype(jnp.int32),axis=0
        ) for i in range(num_qual) 
    ])
    
    
    if len(quant_index) > 0:
        lengthscale = numpyro.sample(
            'covar_module.base_kernel.kernels.1.raw_lengthscale',
            MollifiedUniform(math.log(0.1),math.log(10)).expand([1,len(quant_index)])
        )

        x2_quant = x[:,quant_index]/jnp.exp(lengthscale)
        x2 = jnp.column_stack([x2,x2_quant])
    
    # compute kernel
    k = jnp.exp(outputscale)*cov_map(rbfkernel,x2)
    # add noise and jitter
    k += (jnp.exp(noise)+jitter)*jnp.eye(x.shape[0])
    
    # sample Y according to the standard gaussian process formula
    numpyro.sample(
        "y",
        dist.MultivariateNormal(loc=mean*jnp.ones(x.shape[0]), covariance_matrix=k),
        obs=y,
    )

def numpyro_fitc_lvgp(
    x, y, qual_index, quant_index, num_levels_per_var,
    quant_inducing, qual_weights, approx='FITC', jitter=1e-6
):
    """NumPyro probabilistic program for Sparse Latent Variable GP (FITC/VFE).

    Defines the Bayesian model for sparse LVGP using inducing point approximations.
    Supports both FITC (Fully Independent Training Conditional) and VFE (Variational
    Free Energy) approximations for scalability to large datasets.

    Args:
        x (jax.numpy.ndarray): Training inputs with shape (n, d).
        y (jax.numpy.ndarray): Training targets with shape (n,).
        qual_index (list of int): Column indices for categorical variables.
        quant_index (list of int): Column indices for quantitative variables.
        num_levels_per_var (list of int): Number of levels for each categorical variable.
        quant_inducing (jax.numpy.ndarray): Inducing point locations for quantitative
            inputs with shape (m, len(quant_index)).
        qual_weights (list of jax.numpy.ndarray): Weights for categorical inducing points.
            Each array has shape (m, num_levels) for that categorical variable.
        approx (str, optional): Approximation type, either 'FITC' or 'VFE'.
            Defaults to 'FITC'.
        jitter (float, optional): Numerical stability constant. Defaults to 1e-6.

    Note:
        This is an internal function used by :func:`run_hmc_numpyro`.
        Do not call directly. The sparse approximation uses a low-rank plus diagonal
        covariance structure for efficient computation. For VFE, an additional
        trace term is included via ``numpyro.factor``. Prior specifications:
            - Same latent variable priors as :func:`numpyro_lvgp`
            - Inducing point locations are fixed (not sampled)
    """
    mean = numpyro.sample('mean_module.raw_constant',dist.Normal(0,1))
    outputscale = numpyro.sample("covar_module.raw_outputscale", dist.Normal(0.0, 1))
    noise = numpyro.sample(
        "raw_noise",
        ExpHalfCauchy(1e-2).expand([1])
    )

    num_qual = len(qual_index)

    precisions = [
        numpyro.sample(
            'lv_mapping_layers.%d.raw_precision'%i,
            dist.TransformedDistribution(
                dist.Gamma(2.,1.), 
                dist.transforms.ExpTransform().inv
            ).expand([1])
        ) for i in range(num_qual)
    ]

    raw_latents =[
        numpyro.sample(
            'lv_mapping_layers.%d.raw_latents'%i,
            dist.Normal(0,1).expand([num_levels_per_var[i],2])
        ) for i in range(num_qual)
    ]

    
    latents = ([
        translate_and_rotate(
            raw_latents[i]/jnp.sqrt(jnp.exp(precisions[i]))/math.sqrt(num_levels_per_var[i])
        ) for i in range(num_qual)
    ])
    
    x2 = jnp.column_stack([
        jnp.take(
            latents[i],x[:,qual_index[i]].astype(jnp.int32),axis=0
        ) for i in range(num_qual) 
    ])

    inducing_points = jnp.column_stack([
        qual_weights[i]@latents[i] for i in range(num_qual)
    ])
    
    
    if len(quant_index) > 0:
        lengthscale = numpyro.sample(
            'covar_module.base_kernel.kernels.1.raw_lengthscale',
            MollifiedUniform(math.log(0.1),math.log(10)).expand([1,len(quant_index)])
        )

        x2_quant = x[:,quant_index]/jnp.exp(lengthscale)
        x2 = jnp.column_stack([x2,x2_quant])
        inducing_points = jnp.column_stack([
            inducing_points,quant_inducing/jnp.exp(lengthscale)
        ])

    
    M = inducing_points.shape[0]
    
    Kuu = jnp.exp(outputscale)*cov_map(rbfkernel,inducing_points)
    Kuu += 1e-6*jnp.eye(M) # adding a small jitter term for stability
    Luu = linalg.cholesky(Kuu,lower=True)
    Kuf = jnp.exp(outputscale)*cov_map(rbfkernel,inducing_points,x2)
    W = linalg.solve_triangular(Luu, Kuf,lower=True).transpose()

    Kffdiag = jnp.exp(outputscale)
    Qffdiag = jnp.power(W,2).sum(axis=-1)
    if approx == 'FITC':
        D =  jnp.exp(noise)+jitter+ Kffdiag - Qffdiag
    else:
        D = jnp.repeat(jnp.exp(noise)+jitter,x.shape[0])
        trace_term = (Kffdiag-Qffdiag).sum()/(jnp.exp(noise)+jitter)
        numpyro.factor('vfe_loss_term',-jnp.clip(trace_term,0)/2.)
    
    # sample Y according to the FITC gaussian process formula
    numpyro.sample(
        "y",
        dist.LowRankMultivariateNormal(
            loc=mean*jnp.ones(x.shape[0]), 
            cov_factor=W, 
            cov_diag=D
        ),
        obs=y,
    )


