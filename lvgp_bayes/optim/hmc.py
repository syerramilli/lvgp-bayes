import warnings
import torch
import numpy as np
import gpytorch
import pyro
import gpytorch.settings as gptsettings
from gpytorch.utils.errors import NanError,NotPSDError
from pyro.infer.mcmc import MCMC,NUTS
from pyro.infer.autoguide import init_to_value
from typing import Optional, Dict, Tuple
from collections import OrderedDict
from copy import deepcopy

def pyro_model(model:gpytorch.models.ExactGP):
    sampled_model = model.pyro_sample_from_prior()
    output = sampled_model.likelihood(sampled_model(*model.train_inputs))
    pyro.sample("obs",output,obs=model.train_targets)
    return model.train_targets

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
    model:gpytorch.models.ExactGP,
    init_values:Dict[str,torch.Tensor],
    num_samples:int=1000,
    warmup_steps:int=1000,
    step_size:float=0.1,
    disable_progbar:bool=True,
) -> Tuple[Dict,float]:
    
    hmc_kernel = NUTS(
        pyro_model,
        step_size=step_size,
        adapt_step_size=True,
        max_tree_depth=4,
        init_strategy=init_to_value(values=init_values)
    )
    mcmc = MCMC(
        kernel=hmc_kernel,
        num_samples=num_samples,
        warmup_steps=warmup_steps,
        disable_progbar=disable_progbar,
        num_chains=1
    )
    
    # run mcmc
    mcmc.run(model=model)
    return mcmc

def run_hmc(
    model:gpytorch.models.ExactGP,
    num_samples:int=1000,
    warmup_steps:int=500,
    num_model_samples:int=100,
    step_size:float=0.1,
    disable_progbar:bool=True,
    ):

    # init_values_list = [{} for _ in range(num_chains)]
    # for name,module,prior,closure,_ in model.named_priors():
    #     # first chain is initialized from the current state (preferably the MAP)
    #     init_values_list[0][name] = closure(module).detach().clone()
    #     if num_chains > 1:
    #         # the remaining chains are initialized from some random sample drawn from the prior
    #         for i in range(num_chains-1):
    #             init_values_list[i+1][name] = prior.expand(closure(module).shape).sample()

    
    # set_loky_pickler('dill')
    # mcmc_chains = Parallel(n_jobs=num_jobs,verbose=0)(
    #     delayed(_single_chain_hmc)(
    #         model,init_values,num_samples,warmup_steps,step_size,disable_progbar
    #     ) for init_values in init_values_list
    # )

    # samples_list = [deepcopy(mcmc_chain.get_samples()) for mcmc_chain in mcmc_chains]
    # samples = {}
    # for k in samples_list[0].keys():
    #     v = torch.stack([samples_list[i][k] for i in range(len(samples_list))])
    #     samples[k] = v.reshape((-1,) + v.shape[2:])
    
    init_values  = {}
    for name,module,_,closure,_ in model.named_priors():
        # chain is initialized from the current state (preferably the MAP)
        init_values[name] = closure(module).detach().clone()
    
    flag = False
    while True:
        try:
            with gptsettings.fast_computations(log_prob=False),gptsettings.cholesky_jitter(1e-4):
                mcmc_run = _single_chain_hmc(
                    model=model,
                    init_values=init_values,
                    num_samples=num_samples,
                    warmup_steps=warmup_steps,
                    step_size=step_size,
                    disable_progbar=disable_progbar
                )

            flag = True

        except Exception as e:
            if isinstance(e,NotPSDError) or isinstance(e, NanError):
                warnings.warn(
                    'Chain has reached a point where the model fit is unstable! Restarting MCMC from a new point!',
                    RuntimeWarning
                )

                init_values  = {}
                for name,module,prior,closure,_ in model.named_priors():
                    # chain is initialized from the current state (preferably the MAP)
                    init_values[name] = prior.expand(closure(module).shape).sample()
            else:
                raise
        
        if flag:
            break
        
    samples = deepcopy(mcmc_run.get_samples())
    model.pyro_load_from_samples(get_samples(samples,num_model_samples))

    return mcmc_run