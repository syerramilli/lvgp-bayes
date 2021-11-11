import re
import torch
import numpy as np
import emcee
import gpytorch

from collections import OrderedDict
from typing import Optional,Dict

def load_sample(
    model:gpytorch.models.ExactGP,
    samples_dict:Dict
) -> None:
    '''
    Changes model hyperparameters to those in samples_dict (in-place)
    '''
    model._strict(False)
    for name,_,closure,setting_closure in model.named_priors():
        if not closure().requires_grad:
            continue
        param = samples_dict[name]
        if re.match('.*outputscale.*',name):
             # to ensure batch size is 
             param = param[0]
        setting_closure(param)
    model._strict(True)

class EnsembleMCMC:
    '''
    Uses the emcee package for MCMC instead of HMC/NUTS from pyro
    Requires a lot less memory
    '''
    def __init__(
        self,
        model:gpytorch.models.ExactGP,
        burnin:int=1000,
        chain_length:int=1000,
        p0:Optional[np.ndarray]=None)->None:

        self.model = model
        self.n_dim = 0
        self.named_dims = {}
        self.named_priors = OrderedDict()
        self.param_shapes = OrderedDict()
        self.named_num_params = {}
        

        for name,prior,closure,_ in self.model.named_priors():
            if not closure().requires_grad:
                continue
            num_params = closure().numel()
            self.named_num_params[name] = num_params
            self.param_shapes[name] = closure().shape
            self.named_dims[name] =  self.n_dim+np.arange(num_params)
            self.named_priors[name] = prior
            self.n_dim += num_params
        
        self.n_hypers = max(2*self.n_dim,20)
        if self.n_hypers % 2 == 1:
            self.n_hypers += 1
        
        self.burnin = burnin
        self.num_steps = chain_length + burnin
        self.p0 = p0
        if self.p0 is None:
            self.p0 = self.sample_from_prior(self.n_hypers)
        
        # define ensemble sampler
        self.sampler = emcee.EnsembleSampler(
            nwalkers=self.n_hypers,
            ndim = self.n_dim,
            log_prob_fn=self.log_posterior,
            moves = emcee.moves.DEMove(),
        )
    
    def run(self,progress=False):
        with torch.no_grad():
            pos,_,_ = self.sampler.run_mcmc(self.p0,self.num_steps,progress=progress)
            self.p0 = pos
            self.model.pyro_load_from_samples(self.get_samples_dict(self.p0))
        return pos

    def sample_from_prior(self,num_samples):
        samples = []
        for name,prior in self.named_priors.items():
            if (self.named_num_params[name] > 1) and  (len(prior.event_shape) > 1):
                sample_param = prior.sample((self.n_hypers,)).view(self.n_hypers,-1)
            else:
                sample_param = prior.sample((self.n_hypers,self.named_num_params[name]))
                if sample_param.dim() < 2:
                    sample_param = sample_param.view(-1,1)
            samples.append(sample_param)
        return torch.cat(samples,dim=-1).numpy()

    def get_samples_dict(self,samples:np.ndarray)->Dict:
        params = {}
        for name,dims in self.named_dims.items():
            params[name] = (
                torch.from_numpy(samples[...,dims])
                .reshape(-1,*self.param_shapes[name])
            )
        return params
    
    def log_posterior(self,x):
        log_prior_prob = 0
        params = {}
        for name,prior in self.named_priors.items():
            params[name] = torch.from_numpy(x[self.named_dims[name]])
            if len(self.param_shapes[name])>0:
                params[name] = params[name].reshape(*self.param_shapes[name])
            tmp =  prior.log_prob(params[name]).sum().item()
            if np.isinf(tmp) or np.isnan(tmp):
                return -np.inf
            log_prior_prob += tmp
            
        load_sample(self.model,params)
        del params
        with gpytorch.settings.fast_computations(log_prob=False):
            output_dist = self.model(self.model.train_inputs[0])
            output = self.model.likelihood(output_dist)
            log_lik = output.log_prob(self.model.train_targets)
            #log_lik = self.mll(output,self.mll.model.train_targets).item()
        
        return log_prior_prob + log_lik