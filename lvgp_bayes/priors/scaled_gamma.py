import torch
import torch.distributions as dist
#from torch.nn import Module as TModule
from torch.distributions.utils import broadcast_all
from gpytorch.priors import Prior

class ScaledGamma(dist.TransformedDistribution):
    r"""
    Creates an inverse-gamma distribution parameterized by
    `concentration` and `rate`.

        X ~ Gamma(concentration, rate)
        Y = scale*X ~ ScaledGamma(scale,concentration, rate)

    :param torch.Tensor concentration: the concentration parameter (i.e. alpha).
    :param torch.Tensor rate: the rate parameter (i.e. beta).
    """
    def __init__(self, scale,concentration, rate):
        scale,concentration,rate = broadcast_all(scale,concentration,rate)
        base_dist = dist.Gamma(concentration,rate)
        super(ScaledGamma, self).__init__(
            base_dist,
            dist.AffineTransform(0, scale)
        )

class ScaledGammaPrior(Prior, ScaledGamma):
    """
    Scaled Gamma PRIOR
    """

    def __init__(self, scale,concentration,rate):
        #TModule.__init__(self)
        ScaledGamma.__init__(self, scale,concentration,rate)
        self._transform = None

    def expand(self, batch_shape):
        batch_shape = torch.Size(batch_shape)
        return ScaledGammaPrior(
            self.transforms[0].scale.expand(batch_shape),
            self.base_dist.concentration.expand(batch_shape), 
            self.base_dist.rate.expand(batch_shape)
        )