import warnings
import torch
import numpy as np
import gpytorch
import pyro
import gpytorch.settings as gptsettings
from gpytorch.utils.errors import NanError,NotPSDError
from pyro.infer.mcmc import MCMC,NUTS
from pyro.infer.autoguide import init_to_value

from ..models import GPR,LVGPR
from ..models.orth_lvgp import OrthLVGPR
from ._pyro_models import pyro_lvgp,pyro_gp,pyro_orth_lvgp

from joblib import Parallel,delayed
from joblib.externals.loky import set_loky_pickler
from typing import Optional, Dict, Tuple
from collections import OrderedDict
from copy import deepcopy


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

def _single_chain_hmc(
    model:GPR,
    init_values:Dict[str,torch.Tensor],
    num_samples:int=1000,
    warmup_steps:int=1000,
    disable_progbar:bool=True,
    max_tree_depth:int=5,
    jit_compile:bool=True,
) -> Tuple[Dict,float]:
    
    kwargs = {
        'x':model.train_inputs[0],
        'y':model.train_targets,
        'jitter':model.likelihood.noise_covar.raw_noise_constraint.lower_bound.item()
    }
    if isinstance(model, LVGPR):
        pyro_model = pyro_lvgp
        with torch.no_grad():
            num_levels_per_var = [layer.raw_latents.shape[0] for layer in model.lv_mapping_layers]
        kwargs.update({
            'qual_index':model.qual_index,
            'quant_index':model.quant_index,
            'num_levels_per_var':num_levels_per_var
        })
    elif isinstance(model, OrthLVGPR):
        pyro_model = pyro_orth_lvgp
        with torch.no_grad():
            num_levels_per_var = [layer.raw_weight.shape[-2]+layer.raw_weight.shape[-1] for layer in model.lv_mapping_layers]
        kwargs.update({
            'qual_index':model.qual_index,
            'quant_index':model.quant_index,
            'num_levels_per_var':num_levels_per_var
        })
    
    else:
        pyro_model = pyro_gp
    
    
    hmc_kernel = NUTS(
        pyro_model,
        step_size=0.1,
        adapt_step_size=True,
        max_tree_depth=max_tree_depth,
        init_strategy=init_to_value(values=init_values),
        full_mass=False,
        jit_compile=jit_compile,
        ignore_jit_warnings=True
    )
    mcmc = MCMC(
        kernel=hmc_kernel,
        num_samples=num_samples,
        warmup_steps=warmup_steps,
        disable_progbar=disable_progbar,
        num_chains=1,
    )
    
    # run mcmc
    mcmc.run(**kwargs)
    return mcmc

def run_hmc_jit(
    model:gpytorch.models.ExactGP,
    num_samples:int=500,
    warmup_steps:int=500,
    num_model_samples:int=100,
    disable_progbar:bool=True,
    num_chains:int=1,
    num_jobs:int=1,
    max_tree_depth:int=5,
    jit_compile:bool=True
):

    init_values_list = [{} for _ in range(num_chains)]
    for name,module,prior,closure,_ in model.named_priors():
        for i in range(num_chains):
            init_values_list[i][name[:-6]] = prior.expand(closure(module).shape).sample()

    set_loky_pickler('dill')
    mcmc_runs = Parallel(n_jobs=num_jobs,verbose=0)(
        delayed(_single_chain_hmc)(
            model,init_values,num_samples,warmup_steps,disable_progbar,max_tree_depth,jit_compile
        ) for init_values in init_values_list
    )

    samples_list = [deepcopy(mcmc_run.get_samples()) for mcmc_run in mcmc_runs]
    samples = {}
    for k in samples_list[0].keys():
        v = torch.stack([samples_list[i][k] for i in range(len(samples_list))])
        samples[k] = v.reshape((-1,) + v.shape[2:])
        
    #samples = deepcopy(mcmc_run.get_samples())
    samples = {k:v for k,v in get_samples(samples,num_model_samples).items()}
    model.train_inputs = tuple(tri.unsqueeze(0).expand(num_model_samples, *tri.shape) for tri in model.train_inputs)
    model.train_targets = model.train_targets.unsqueeze(0).expand(num_model_samples, *model.train_targets.shape)

    state_dict = deepcopy(model.state_dict())
    state_dict.update(samples)

    # Load parameters without standard shape checking.
    model.load_strict_shapes(False)
    model.load_state_dict(state_dict)

    # model.pyro_load_from_samples(get_samples(samples,num_model_samples))

    return mcmc_runs