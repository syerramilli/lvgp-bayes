import torch 
import torch.distributions as dist
from gpytorch.priors import Prior

class ExpGamma(dist.TransformedDistribution):
    r"""
    Creates an inverse-gamma distribution parameterized by
    `concentration` and `rate`.

        exp(X) ~ Gamma(concentration, rate)
        
    :param torch.Tensor concentration: the concentration parameter (i.e. alpha).
    :param torch.Tensor rate: the rate parameter (i.e. beta).
    """
    def __init__(self,concentration, rate):
        super(ExpGamma, self).__init__(
            dist.Gamma(concentration,rate),
            dist.ExpTransform().inv
        )

class ExpGammaPrior(Prior, ExpGamma):
    """
    ExpGamma Prior
    """
    def __init__(self,concentration,rate):
        #TModule.__init__(self)
        ExpGamma.__init__(self,concentration,rate)
        self._transform = None

    def expand(self, batch_shape):
        batch_shape = torch.Size(batch_shape)
        return ExpGammaPrior(
            self.base_dist.concentration.expand(batch_shape), 
            self.base_dist.rate.expand(batch_shape)
        )