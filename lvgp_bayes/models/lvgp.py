import torch
import math
import gpytorch
from gpytorch.constraints import Positive
from gpytorch.priors import NormalPrior
from gpytorch.distributions import MultivariateNormal
from .gpregression import GPR
from .. import kernels
from ..priors.exp_gamma import ExpGammaPrior
from ..priors.mollified_uniform import MollifiedUniformPrior
from typing import List,Optional


class LVMapping(gpytorch.Module):
    """Latent variable mapping. 
    
    Maps the levels of a qualitative variable onto a latent numerical space. This is implemented 
    in the form of a lookup table similar to `torch.nn.Embedding`, although the parameterization
    is somewhat different. The parameterization ensures that the mapping is not invariant to 
    translation, and rotation. However, it is still invariant to reflection. 

    :note: Binary categorical variables should not be supplied. There is no benefit from applying a 
        latent variable treatment for such variables. Instead, treat them as numerical inputs.

    :param num_levels: The number of levels for the categorical variable
    :type num_levels: int
    
    :param lv_dim: The dimension of the latent variable space. This needs to be greater than 1
        and can atmost be `num_levels`-1. 
    :type lv_dim: int

    :param batch_shape: not currently supported
    """
    def __init__(
        self,
        num_levels:int,
        lv_dim:int,
        batch_shape=torch.Size()
    ) -> None:
        super().__init__()
        self.batch_shape = batch_shape

        if num_levels == 1:
            raise ValueError('Categorical variable has only one level!')
        elif num_levels == 2:
            raise ValueError('Binary categorical variables should not be supplied')

        if lv_dim == 1:
            raise RuntimeWarning('1D latent variables are difficult to optimize!')
        elif lv_dim > num_levels - 1:
            lv_dim = num_levels-1
            raise RuntimeWarning(
                'The LV dimension can atmost be num_levels-1. '
                'Setting it to %s in place of %s' %(num_levels-1,lv_dim)
            )
        
        self.register_buffer('num_levels',torch.tensor(num_levels))
        self.register_buffer('lv_dim',torch.tensor(lv_dim))
        self.register_parameter(
            name='raw_latents',
            parameter=torch.nn.Parameter(
                torch.randn(*batch_shape,num_levels,lv_dim)
            )
        )
        self.register_prior(
            name='latents_prior',
            prior=gpytorch.priors.NormalPrior(0.,1.),
            param_or_closure='raw_latents'
        )

        self.register_parameter(
            'raw_precision',
            parameter = torch.nn.Parameter(torch.zeros(1))
        )
        self.register_constraint(
            param_name='raw_precision',
            constraint=gpytorch.constraints.Positive(transform=torch.exp,inv_transform=torch.log)
        )
        self.register_prior(
            name='precision_prior',
            prior=ExpGammaPrior(2.,2.),
            param_or_closure='raw_precision'
        )
        
    @property
    def precision(self):
        return self.raw_precision_constraint.transform(self.raw_precision)
    
    def _set_precision(self,value):
        raw_value = (
            self.raw_precision_constraint
            .inverse_transform(value.to(self.raw_precision))
        )
        self.initialize(**{'raw_precision':raw_value})

    @property
    def latents(self):
        batch_size = torch.Size([]) if self.raw_precision.ndim == 1 else torch.Size([self.raw_precision.numel()])
        return 1/self.precision.view(*batch_size,1,1).sqrt()*self.raw_latents
        
    def forward(self,x:torch.LongTensor)->torch.Tensor:
        """Map the levels of the qualitative factor onto the latent variable space.

        :param x: 1D tensor of levels (which need to be encoded as integers) of size N
        :type x: torch.LongTensor

        :returns: a N x lv_dim tensor
        :rtype: torch.Tensor
        """
        if self.latents.ndim == 3:
            return torch.stack([
                torch.nn.functional.embedding(x[i,:],self.latents[i,...]) \
                    for i in range(self.latents.shape[0])
            ])
        return torch.nn.functional.embedding(x,self.latents)

class LVGPR(GPR):
    """The latent variable GP regression model which extends GPs to handle categorical inputs.

    This is based on the work of `Zhang et al. (2019)`_. LVGPR first projects each categorical input 
    onto a numerical latent variable space, which can then be used with standard GP kernels for numerical 
    inputs. These latent variables are jointly estimated along with the other GP hyperparameters.

    :note: Binary categorical variables should not be treated as qualitative inputs. There is no 
        benefit from applying a latent variable treatment for such variables. Instead, treat them
        as numerical inputs.

    :param train_x: The training inputs (size N x d). Qualitative inputs needed to be encoded as 
        integers 0,...,L-1 where L is the number of levels. For best performance, scale the 
        numerical variables to the unit hypercube.
    :type train_x: torch.Tensor
    :param train_y: The training targets (size N)
    :type train_y: torch.Tensor
    :param qual_index: List specifying the indices for the qualitative inputs in the data. This
        list cannot be empty.
    :type qual_index: List[int]
    :param quant_index: List specifying the indices for the quantitative inputs in the data.
    :type quant_index: List[int]
    :param num_levels_per_var: List specifying the number of levels for each qualitative variable.
        The order should correspond to the one specified in `qual_index`. This list cannot be empty.
    :type num_levels_per_var: List[int]
    :param lv_dim: The dimension of the latent variable space for each qualitative input. Defaults to 2.
    :type lv_dim: int
    :param quant_correlation_class: A string specifying the kernel for the quantitative inputs. Needs
        to be one of the following strings - 'RBFKernel' (radial basis kernel), 'Matern52Kernel' (twice 
        differentiable Matern kernel), 'Matern32Kernel' (first order differentiable Matern
        kernel). The generate kernel uses a separate lengthscale for each input variable. Defaults to
        'RBFKernel'.
    :type quant_correlation_class: str, optional
    :param noise: The (initial) noise variance.
    :type noise: float, optional
    :param fix_noise: Fixes the noise variance at the current level if `True` is specifed.
        Defaults to `False`
    :type fix_noise: bool, optional
    :param lb_noise: Lower bound on the noise variance. Setting a higher value results in
        more stable computations, when optimizing noise variance, but might reduce 
        prediction quality. Defaults to 1e-6
    :type lb_noise: float, optional

    .. _Zhang et al. (2019):
        https://doi.org/10.1080/00401706.2019.1638834
    """
    def __init__(
        self,
        train_x:torch.Tensor,
        train_y:torch.Tensor,
        qual_index:List[int],
        quant_index:List[int],
        num_levels_per_var:List[int],
        lv_dim:int=2,
        quant_correlation_class:str='RBFKernel',
        noise:float=1e-4,
        fix_noise:bool=False,
        lb_noise:float=1e-6,
    ) -> None:

        qual_kernel = kernels.RBFKernel(
            active_dims=torch.arange(len(qual_index)*lv_dim)
        )
        qual_kernel.initialize(**{'lengthscale':1.0})
        qual_kernel.raw_lengthscale.requires_grad_(False)

        if len(quant_index) == 0:
            correlation_kernel = qual_kernel
        else:
            try:
                quant_correlation_class = getattr(kernels,quant_correlation_class)
            except:
                raise RuntimeError(
                    "%s not an allowed kernel" % quant_correlation_class
                )
            quant_kernel = quant_correlation_class(
                ard_num_dims=len(quant_index),
                active_dims=len(qual_index)*lv_dim+torch.arange(len(quant_index)),
                lengthscale_constraint=Positive(transform=torch.exp,inv_transform=torch.log)
            )
            quant_kernel.register_prior(
                'lengthscale_prior',MollifiedUniformPrior(math.log(0.05), math.log(10)),'raw_lengthscale'
            )
            correlation_kernel = qual_kernel*quant_kernel

        super(LVGPR,self).__init__(
            train_x=train_x,train_y=train_y,
            correlation_kernel=correlation_kernel,
            noise=noise,fix_noise=fix_noise,lb_noise=lb_noise
        )

        # register index and transforms
        self.register_buffer('quant_index',torch.tensor(quant_index))
        self.register_buffer('qual_index',torch.tensor(qual_index))

        # latent variable mapping
        self.lv_mapping_layers = torch.nn.ModuleList([
            LVMapping(num_levels,lv_dim) \
                for k,num_levels in enumerate(num_levels_per_var)
        ])
    
    def forward(self,x:torch.Tensor) -> MultivariateNormal:
        embeddings = []
        for i,e in enumerate(self.lv_mapping_layers):
            embeddings.append(e(x[...,self.qual_index[i]].long()))

        embeddings = torch.cat(embeddings,-1)
        if len(self.quant_index) > 0:
            x = torch.cat([embeddings,x[...,self.quant_index]],-1)
        else:
            x = embeddings

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x,covar_x)
    
    def named_hyperparameters(self):
        """Return all hyperparameters other than the latent variables

        This method is useful when different learning rates to the latent variables. To 
        include the latent variables along with others use `.named_parameters` method
        """
        for name, param in self.named_parameters():
            if "lv_mapping" not in name:
                yield name, param
    
    def to_pyro_random_module(self):
        new_module = super().to_pyro_random_module()
        # some modules are not registered as Pyro modules
        if isinstance(self.covar_module.base_kernel,gpytorch.kernels.ProductKernel):
            new_module.covar_module.base_kernel.kernels[1] = \
                new_module.covar_module.base_kernel.kernels[1].to_pyro_random_module()
        for i,layer in enumerate(new_module.lv_mapping_layers):
            new_module.lv_mapping_layers[i]= layer.to_pyro_random_module()

        return new_module