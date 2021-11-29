import torch
import gpytorch
import math
from gpytorch.models import ExactGP
from gpytorch.priors import NormalPrior
from gpytorch.constraints import GreaterThan,Positive
from gpytorch.distributions import MultivariateNormal
from .. import kernels
from ..priors.horseshoe import ExpHalfHorseshoePrior
from ..priors.exp_gamma import ExpGammaPrior
from ..priors.mollified_uniform import MollifiedUniformPrior
from typing import List,Tuple,Union

class GPR(ExactGP):
    """Standard GP regression module for numerical inputs

    :param train_x: The training inputs (size N x d). All input variables are expected
        to be numerical. For best performance, scale the variables to the unit hypercube.
    :type train_x: torch.Tensor
    :param train_y: The training targets (size N)
    :type train_y: torch.Tensor
    :param correlation_kernel: Either a `gpytorch.kernels.Kernel` instance or one of the 
        following strings - 'RBFKernel' (radial basis kernel), 'Matern52Kernel' (twice 
        differentiable Matern kernel), 'Matern32Kernel' (first order differentiable Matern
        kernel). If the former is specified, any hyperparameters to be estimated need to have 
        associated priors for multi-start optimization. If the latter is specified, then 
        the kernel uses a separate lengthscale for each input variable.
    :type correlation_kernel: Union[gpytorch.kernels.Kernel,str]
    :param noise: The (initial) noise variance.
    :type noise: float, optional
    :param fix_noise: Fixes the noise variance at the current level if `True` is specifed.
        Defaults to `False`
    :type fix_noise: bool, optional
    :param lb_noise: Lower bound on the noise variance. Setting a higher value results in
        more stable computations, when optimizing noise variance, but might reduce 
        prediction quality. Defaults to 1e-6
    :type lb_noise: float, optional
    """
    def __init__(
        self,
        train_x:torch.Tensor,
        train_y:torch.Tensor,
        correlation_kernel,
        noise:float=1e-4,
        fix_noise:bool=False,
        lb_noise:float=1e-6,
    ) -> None:
        # initializing likelihood
        noise_constraint=GreaterThan(lb_noise,transform=torch.exp,inv_transform=torch.log)
        likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=noise_constraint)

        # standardizing the response variable
        y_mean,y_std = train_y.mean(),train_y.std()
        train_y_sc = (train_y-y_mean)/y_std

        # initializing ExactGP
        super().__init__(train_x,train_y_sc,likelihood)

        # registering mean and std of the raw response
        self.register_buffer('y_mean',y_mean)
        self.register_buffer('y_std',y_std)

        # initializing and fixing noise
        if noise is not None:
            self.likelihood.initialize(noise=noise)
        
        self.likelihood.register_prior('noise_prior',ExpHalfHorseshoePrior(0.01,lb_noise),'raw_noise')
        if fix_noise:
            self.likelihood.raw_noise.requires_grad_(False)
        
        # Modules
        self.mean_module = gpytorch.means.ConstantMean(prior=NormalPrior(0.,1.))
        if isinstance(correlation_kernel,str):
            try:
                correlation_kernel_class = getattr(kernels,correlation_kernel)
                correlation_kernel = correlation_kernel_class(
                    ard_num_dims = self.train_inputs[0].size(1),
                    lengthscale_constraint=Positive(transform=torch.exp,inv_transform=torch.log),
                )
                correlation_kernel.register_prior(
                    'lengthscale_prior',MollifiedUniformPrior(math.log(0.1),math.log(10)),'raw_lengthscale'
                )
            except:
                raise RuntimeError(
                    "%s not an allowed kernel" % correlation_kernel
                )
        elif not isinstance(correlation_kernel,gpytorch.kernels.Kernel):
            raise RuntimeError(
                "specified correlation kernel is not a `gpytorch.kernels.Kernel` instance"
            )

        self.covar_module = kernels.ScaleKernel(
            base_kernel = correlation_kernel,
            outputscale_constraint=Positive(transform=torch.exp,inv_transform=torch.log),
        )
        # register priors
        self.covar_module.register_prior(
            'outputscale_prior',NormalPrior(0.,1.),'raw_outputscale'
        )
    
    def forward(self,x:torch.Tensor)->MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x,covar_x)
    
    def predict(
        self,x:torch.Tensor,return_std:bool=False,include_noise:bool=False
    )-> Union[torch.Tensor,Tuple[torch.Tensor]]:
        """Returns the predictive mean, and optionally the standard deviation at the given points

        :param x: The input variables at which the predictions are sought. 
        :type x: torch.Tensor
        :param return_std: Standard deviation is returned along the predictions  if `True`. 
            Defaults to `False`.
        :type return_std: bool, optional
        :param include_noise: Noise variance is included in the standard deviation if `True`. 
            Defaults to `False`.
        :type include_noise: bool
        """
        self.eval()
        
        # determine if batched or not
        ndim = self.train_targets.ndim
        if ndim == 1:
            output = self(x)
        else:
            # for batched GPs 
            num_samples = self.train_targets.shape[0]
            output = self(x.unsqueeze(0).repeat(num_samples,1,1))
        
        if return_std and include_noise:
            output = self.likelihood(output)

        out_mean = self.y_mean + self.y_std*output.mean

        # standard deviation may not always be needed
        if return_std:
            out_std = output.variance.sqrt()*self.y_std
            return out_mean,out_std

        return out_mean
    
    def reset_parameters(self) -> None:
        """Reset parameters by sampling from prior
        """
        for _,module,prior,closure,setting_closure in self.named_priors():
            if not closure(module).requires_grad:
                continue
            setting_closure(module,prior.expand(closure(module).shape).sample())
    
    def load_samples(self, samples_dict):
        '''
        Replace the model's learned hyperparameters with samples from a posterior distribution
        Modifying existing method to change only parameters with gradients. Uses Pyro convention
        '''
        # Pyro always puts the samples in the first batch dimension
        num_samples = next(iter(samples_dict.values())).size(0)
        self.train_inputs = tuple(tri.unsqueeze(0).expand(num_samples, *tri.shape) for tri in self.train_inputs)
        self.train_targets = self.train_targets.unsqueeze(0).expand(num_samples, *self.train_targets.shape)
        self._strict(False)
        for name, _, closure, setting_closure in self.named_priors():
            if not closure().requires_grad:
                continue
            # set parameter
            setting_closure(samples_dict[name])

        self._strict(True)