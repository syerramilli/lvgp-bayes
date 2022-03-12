import math
import pyro
import torch
import pyro.distributions as dist

from gpytorch.lazy import lazify
from gpytorch.distributions import MultivariateNormal
from lvgp_bayes.priors import ExpGammaPrior,ExpHalfHorseshoePrior,MollifiedUniformPrior

class MVN(dist.MultivariateNormal):
    def __init__(self,base_dist,added_loss_terms=None):
        self.base_dist = base_dist
    
    def rsample(self,**kwargs):
        return self.base_dist.rsample(**kwargs)
    
    def log_prob(self,value):
        try:
            out= self.base_dist.log_prob(value)
            return out
        except Exception as e:
            raise RuntimeError("singular U")
    
    @property
    def event_shape(self):
        return self.base_dist.event_shape
    
    @property
    def _batch_shape(self):
        return self.base_dist._batch_shape

def rbf_kernel(x1,x2=None):
    if x2 is None:
        x2 = x1
    
    x1_sq = (x1**2).sum(1,keepdim=True)
    x2_sq = (x2**2).sum(1,keepdim=True)
    x1x2 = x1.matmul(x2.t()) 
    r2 = x1_sq + x2_sq.t() - 2*x1x2
    return torch.exp(-0.5*r2.clamp(min=0))

def pyro_gp(x,y,jitter):
    mean = pyro.sample('mean_module.constant',dist.Normal(0,1).expand([1]))
    outputscale = pyro.sample("covar_module.raw_outputscale", dist.Normal(0.0, 1))
    noise = pyro.sample(
        "likelihood.noise_covar.raw_noise",
        ExpHalfHorseshoePrior(0.01,jitter).expand([1])
    )
    
    lengthscale = pyro.sample(
        'covar_module.base_kernel.raw_lengthscale',
        MollifiedUniformPrior(math.log(0.1),math.log(10)).expand([1,x.shape[1]]).to_event(2)
    )

    x2 = x/lengthscale.exp()
    
    # compute kernel
    k = outputscale.exp()*rbf_kernel(x2)
    # add noise and jitter
    k += (noise.exp()+jitter)*torch.eye(x.shape[0])
    
    # sample Y according to the standard gaussian process formula
    pyro.sample(
        "y",
        MVN(
            MultivariateNormal(
                mean*torch.ones(x.shape[0]).to(x), lazify(k.to(x))
            )
        ),
        obs=y,
    )

def pyro_lvgp(x,y,qual_index,quant_index,num_levels_per_var,jitter):
    mean = pyro.sample('mean_module.constant',dist.Normal(0,1).expand([1]))
    outputscale = pyro.sample("covar_module.raw_outputscale", dist.Normal(0.0, 1))
    noise = pyro.sample(
        "likelihood.noise_covar.raw_noise",
        ExpHalfHorseshoePrior(0.01,jitter).expand([1])
    )

    num_qual = len(qual_index)
    
    precisions = [
        pyro.sample('lv_mapping_layers.%d.raw_precision'%i,
                    ExpGammaPrior(2,1).expand([1])
                   )\
        for i in range(num_qual)
    ]
    
    raw_latents = [
        pyro.sample(
            'lv_mapping_layers.%d.raw_latents'%i,
            dist.Normal(0,1).expand([num_levels_per_var[i],2]).to_event(2)
        )\
        for i in range(num_qual)
    ]
    
    x2 = torch.cat([
        (raw_latents[i]/(precisions[i].exp().sqrt()))[x[:,qual_index[i]].long(),:] for i in range(num_qual)
    ],dim=-1)

    if len(quant_index) > 0:
        lengthscale = pyro.sample(
            'covar_module.base_kernel.kernels.1.raw_lengthscale',
            MollifiedUniformPrior(math.log(0.1),math.log(10)).expand([1,len(quant_index)]).to_event(2)
        )

        x2_quant = x[:,quant_index]/lengthscale.exp()
        x2 = torch.cat([x2,x2_quant],dim=-1)
    
    # compute kernel
    k = outputscale.exp()*rbf_kernel(x2)
    # add noise and jitter
    k += (noise.exp()+jitter)*torch.eye(x.shape[0])
    
    # sample Y according to the standard gaussian process formula
    pyro.sample(
        "y",
        MVN(
            MultivariateNormal(
                mean*torch.ones(x.shape[0]).to(x), lazify(k.to(x))
            )
        ),
        obs=y,
    )