import numpy as np
import jax.numpy as jnp
import torch
import jax.random as random
from numpyro.infer import Predictive, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoLowRankMultivariateNormal
from numpyro.optim import Adam

from .numpyro_hmc import (
    numpyro_gpr,
    numpyro_lvgp,
    get_samples
)
from ..models import GPR,LVGPR
from ..models.sparselvgp import SparseLVGPR

from typing import Optional, Dict, Tuple
from collections import OrderedDict
from copy import deepcopy

def run_svi_numpyro(
    model,
    num_steps:int=200,
    num_model_samples:int=100,
    step_size:float=1e-3,
    disable_progbar:bool=True,
    seed=0,
):
    kwargs = {
        'x':jnp.array(model.train_inputs[0].numpy()),
        'y':jnp.array(model.train_targets.numpy()),
        
    }
    
    if isinstance(model, LVGPR):
        numpyro_model = numpyro_lvgp
        with torch.no_grad():
            num_levels_per_var = [layer.raw_latents.shape[0] for layer in model.lv_mapping_layers]
        
        kwargs.update({
            'qual_index':model.qual_index.tolist(),
            'quant_index':model.quant_index.tolist(),
            'num_levels_per_var':num_levels_per_var,
            'jitter':model.likelihood.noise_covar.raw_noise_constraint.lower_bound.item()
        })

    else:
        numpyro_model = numpyro_gpr

        kwargs.update({
            'jitter':model.likelihood.noise_covar.raw_noise_constraint.lower_bound.item(),
            'kernel':model.covar_module.base_kernel.__class__.__name__.lower()
        })

    
    
    rng_key, rng_key_predict = random.split(random.PRNGKey(seed))
    guide = AutoLowRankMultivariateNormal(numpyro_model)
    svi = SVI(
        numpyro_model, 
        guide=guide,optim= Adam(step_size=1e-3), loss=Trace_ELBO()
    )
    svi_result = svi.run(
        rng_key, num_steps=num_steps, 
        progress_bar= (not disable_progbar),
        **kwargs
    )
    predictive = Predictive(guide, params=svi_result.params, num_samples=num_model_samples)
    samples = predictive(rng_key_predict, **kwargs)

    samples = {
        k:torch.from_numpy(np.array(v)).to(model.train_targets) \
            for k,v in samples.items() if k not in ['_auto_latent']
    }

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

    out = {
        'params':svi_result.params,
        'losses':svi_result.losses
    }
    return out