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
    """Compute a covariance matrix from a covariance function and data points.
    Args:
      cov_func: callable function, maps pairs of data points to scalars.
      xs: array of data points, stacked along the leading dimension.
    Returns:
      A 2d array `a` such that `a[i, j] = cov_func(xs[i], xs[j])`.
    """
    if xs2 is None:
        return vmap(lambda x: vmap(lambda y: cov_func(x, y))(xs))(xs)
    else:
        return vmap(lambda x: vmap(lambda y: cov_func(x, y))(xs))(xs2).T
    
def rbfkernel(x1, x2):
    return jnp.exp(-0.5*jnp.sum((x1 - x2)**2))

def matern52kernel(x1,x2):
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
    return Z-Z[...,[0],:]

def translate_and_rotate(Z):
    Zcen = Z-Z[...,[0],:]
    theta = jnp.arctan(Zcen[...,[1],[1]]/Zcen[...,[1],[0]])
    c,s = jnp.cos(theta),jnp.sin(theta)
    R = jnp.vstack([
        jnp.column_stack([c,-s]),
        jnp.column_stack([s,c])
    ])
    return Zcen @ R

class ExpHalfCauchy(dist.TransformedDistribution):
    def __init__(self,scale):
        
        base_dist = dist.HalfCauchy(scale)
        super().__init__(
            base_dist,dist.transforms.ExpTransform().inv
        )

def get_samples(samples,num_samples=None, group_by_chain=False):
    """
    Get samples from the MCMC run

    :param int num_samples: Number of samples to return. If `None`, all the samples
        from an MCMC chain are returned in their original ordering.
    :param bool group_by_chain: Whether to preserve the chain dimension. If True,
        all samples will have num_chains as the size of their leading dimension.
    :return: dictionary of samples keyed by site name.
    """
    if num_samples is not None:
        batch_dim = 0
        sample_tensor = list(samples.values())[0]
        batch_size, device = sample_tensor.shape[batch_dim], sample_tensor.device
        idxs = torch.linspace(0,batch_size-1,num_samples,dtype=torch.long,device=device).flip(0)
        samples = {k: v.index_select(batch_dim, idxs) for k, v in samples.items()}
    return samples

def run_hmc_numpyro(
    model,
    num_samples:int=500,
    warmup_steps:int=500,
    num_model_samples:int=100,
    disable_progbar:bool=True,
    num_chains:int=1,
    num_jobs:int=1,
    max_tree_depth:int=5,
    initialize_from_state:bool=False,
    seed:int=0
):
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
# Numpyro models
########

## Regular GP with RBF kernel
def numpyro_gpr(
    x,y,kernel='rbfkernel',jitter=1e-6
):
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

## LVGP model
def numpyro_lvgp(
    x,y,qual_index,quant_index,num_levels_per_var,jitter=1e-6,alpha=2.,beta=1.
):
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
    x,y,qual_index,quant_index,num_levels_per_var,
    quant_inducing,qual_weights,approx='FITC',jitter=1e-6
):
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


