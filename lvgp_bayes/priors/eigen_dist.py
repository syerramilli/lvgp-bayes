import torch
import math
from gpytorch.priors import Prior
from pyro.distributions import TorchDistribution, constraints,TransformedDistribution
from pyro.distributions.transforms import SoftplusTransform,OrderedTransform,ComposeTransform

orig_to_raw_transform = ComposeTransform([SoftplusTransform().inv,OrderedTransform().inv])

class EigenvaluesDist(TorchDistribution):
    support = constraints.positive_ordered_vector
    def __init__(self,orig_dim,latent_dim):
        super().__init__(validate_args=False,event_shape=torch.Size([1,latent_dim]))
        self.orig_dim = orig_dim
        self.latent_dim = latent_dim
    
    def log_prob(self,x):
        # unnormalized log probability density
        target = -0.5*torch.sum(x**2,dim=-1).flatten() + (self.orig_dim-self.latent_dim-1)*x.log().sum(dim=-1).flatten()
        for i in range(self.latent_dim):
            for j in range(i+1,self.latent_dim):
                target += torch.log(x[...,self.latent_dim-i-1]**2-x[...,self.latent_dim-j-1]**2).flatten()
        
        target += (2*x).log().sum(dim=-1)
        return target
    
    def rsample(self,sample_shape=torch.Size([])):
        # method only serves to provide an initial guess for HMC
        # returns the approximate median
        return math.sqrt(self.orig_dim)*(
            0.9*torch.ones(1,self.latent_dim) + 0.1*torch.arange(self.latent_dim) +\
                torch.sort(0.1*torch.randn(1,self.latent_dim))[0]
        )


class RawEigenvaluesPrior(TransformedDistribution,Prior):
    def __init__(self,orig_dim,latent_dim):
        base_dist = EigenvaluesDist(orig_dim, latent_dim)
        super().__init__(base_dist,orig_to_raw_transform)
        Prior.__init__(self)

    def expand(self,*args):
        return self