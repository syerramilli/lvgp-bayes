import torch
from torch.nn import Module as TModule
from gpytorch.priors import Prior
from torch.distributions import StudentT

class StudentTPrior(Prior, StudentT):
    """
    StudentT prior.
    """
    def __init__(self, df,loc, scale, validate_args=None, transform=None):
        TModule.__init__(self)
        StudentT.__init__(self,df=df,loc=loc, scale=scale, validate_args=validate_args)
        self._transform = transform

    def expand(self, batch_shape):
        batch_shape = torch.Size(batch_shape)
        return StudentTPrior(self.df.expand(batch_shape),self.loc.expand(batch_shape), self.scale.expand(batch_shape))