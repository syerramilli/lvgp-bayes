import torch
import gpytorch
import math

from torch.distributions import LowRankMultivariateNormal
from gpytorch.priors import NormalPrior
from gpytorch.constraints import GreaterThan,Positive
from .lvgp import LVMapping
from .. import kernels
from ..priors.horseshoe import ExpHalfHorseshoePrior
from ..priors.mollified_uniform import MollifiedUniformPrior
from typing import List,Optional,Union,Tuple

def theta_to_weights(raw_theta):
    theta = torch.sigmoid(raw_theta)*math.pi/2
    w = (
        torch.cat([
            torch.ones(*theta.shape[:-1],1),torch.cumprod(theta.sin(),dim=-1)
        ],dim=-1)*torch.cat([
            theta.cos(),torch.ones(*theta.shape[:-1],1)
        ],dim=-1)
    )**2
    return w

class VFEApproxAddedLossTerm(gpytorch.mlls.AddedLossTerm):
    def __init__(self,Kffdiag,Qffdiag,noise):
        self.diag_term = (Kffdiag-Qffdiag).sum()
        self.noise = noise

    def loss(self):
        return -0.5*(self.diag_term.clamp(min=0))/self.noise

class SparseLVGPR(gpytorch.Module):
    def __init__(
        self,
        train_x:torch.Tensor,
        train_y:torch.Tensor,
        num_inducing:int,
        qual_index:List[int],
        quant_index:List[int],
        num_levels_per_var:List[int],
        lv_dim:int=2,
        quant_correlation_class:str='RBFKernel',
        approx:str='VFE',
        noise:float=1e-4,
        lb_noise:float=1e-6,
    ) -> None:

        super().__init__()
        # adding data to model
        self.train_inputs = (train_x,)
        # adding the standardized response variable
        y_mean,y_std = train_y.mean(),train_y.std()
        self.train_targets = (train_y-y_mean)/y_std

        # registering mean and std of the raw response
        self.register_buffer('y_mean',y_mean)
        self.register_buffer('y_std',y_std)

        # noise parameter and prior
        self.register_parameter(
            'raw_noise', torch.nn.Parameter(0.1*torch.randn(1))
        )
        self.register_constraint(
            'raw_noise', 
            GreaterThan(lb_noise,transform=torch.exp,inv_transform=torch.log)
        )
        self.register_prior('raw_noise_prior', ExpHalfHorseshoePrior(1e-2, 1e-6), 'raw_noise')

        # mean module
        self.mean_module = gpytorch.means.ConstantMean(prior=NormalPrior(0.,1.))

        # correlation kernel
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
                lengthscale_constraint=Positive(transform=torch.exp,inv_transform=torch.log),
            )
            quant_kernel.register_prior(
                'raw_lengthscale_prior',MollifiedUniformPrior(math.log(0.1),math.log(10)),'raw_lengthscale',
            )
            correlation_kernel = qual_kernel*quant_kernel
            
        
        # covariance modules
        self.covar_module = kernels.ScaleKernel(
            base_kernel = correlation_kernel,
            outputscale_constraint=Positive(transform=torch.exp,inv_transform=torch.log),
        )

        self.covar_module.register_prior(
            'raw_outputscale_prior',NormalPrior(0.,1.),'raw_outputscale'
        )

        # register index and transforms
        self.register_buffer('quant_index',torch.tensor(quant_index))
        self.register_buffer('qual_index',torch.tensor(qual_index))

        # latent variable mapping
        self.lv_mapping_layers = torch.nn.ModuleList([
            LVMapping(num_levels,lv_dim) \
                for k,num_levels in enumerate(num_levels_per_var)
        ])

        # inducing points
        for k,num_levels in enumerate(num_levels_per_var):
            self.register_parameter(
                'raw_thetas%d'%k, 
                torch.nn.Parameter(0.1*torch.randn(num_inducing,num_levels-1))
            )

        self.register_parameter(
            'raw_quant_inducing', 
            torch.nn.Parameter(0.1*torch.randn(num_inducing,len(quant_index)))
        )

        # type of approximation
        self.approx = approx
        if self.approx == 'VFE':
            self.register_added_loss_term("vfe_loss_term")
    
    def inducing_points(self):
        out = torch.cat([
            theta_to_weights(
                getattr(self,'raw_thetas%d'%k)
            ) @ layer.latents for k,layer in enumerate(self.lv_mapping_layers)
        ],dim=-1)

        out = torch.cat([
            out,torch.sigmoid(self.raw_quant_inducing)
        ],dim=-1)
        
        # if self.raw_noise.ndim > 1:
        #     return out.unsqueeze(0).repeat(self.raw_noise.shape[0],1,1)
        
        return out

    def transform_inputs(self,x:torch.Tensor) -> torch.Tensor:
        embeddings = []
        for i,e in enumerate(self.lv_mapping_layers):
            embeddings.append(e(x[...,self.qual_index[i]].long()))

        embeddings = torch.cat(embeddings,-1)
        if len(self.quant_index) > 0:
            x = torch.cat([embeddings,x[...,self.quant_index]],-1)
        else:
            x = embeddings
        
        return x
    
    # prior
    def forward(self,x:torch.Tensor) -> LowRankMultivariateNormal:
        # W = (inv(Luu) @ Kuf).T
        # Qff = Kfu @ inv(Kuu) @ Kuf = W @ W.T
        # y_cov = Qff + diag(Kff - Qff) + noise
        x = self.transform_inputs(x)
        Xu = self.inducing_points()
        N = x.shape[0]
        M = Xu.shape[0]
        
        Kuu = self.covar_module(Xu).evaluate()
        Kuu.view(-1)[::M+1] += 1e-6 # adding a small jitter term for stability
        Luu = torch.linalg.cholesky(Kuu)
        Kuf = self.covar_module(Xu,x).evaluate()
        W = torch.triangular_solve(Kuf, Luu,upper=False)[0].transpose(-2,-1)
        Kffdiag = self.covar_module(x,diag=True)
        Qffdiag = W.pow(2).sum(dim=-1)

        if self.approx=='FITC':
            D = self.raw_noise_constraint.transform(self.raw_noise) + Kffdiag-Qffdiag
        else:
            noise = self.raw_noise_constraint.transform(self.raw_noise)
            D = noise.repeat(N)
            # loss term
            self.update_added_loss_term('vfe_loss_term',VFEApproxAddedLossTerm(Kffdiag, Qffdiag, noise))
        
        mean_x = self.mean_module(x)
        return LowRankMultivariateNormal(mean_x,W,D)

    def __call__(self,*args,**kwargs):
        train_inputs = list(self.train_inputs) if self.train_inputs is not None else []
        inputs = [i.unsqueeze(-1) if i.ndimension() == 1 else i for i in args]

        # Training mode: optimizing
        if self.training:
            if self.train_inputs is None:
                raise RuntimeError(
                    "train_inputs, train_targets cannot be None in training mode. "
                    "Call .eval() for prior predictions, or call .set_train_data() to add training data."
                )

            if not all(torch.equal(train_input, input) for train_input, input in zip(train_inputs, inputs)):
                raise RuntimeError("You must train on the training inputs!")

            res = self.forward(*inputs,**kwargs)
            return res
        # posterior mode
        else:
            # TODO: cache some of the computations
            # W = inv(Luu) @ Kuf
            # Ws = inv(Luu) @ Kus
            # D as in self.model()
            # K = I + W @ inv(D) @ W.T = L @ L.T
            # S = inv[Kuu + Kuf @ inv(D) @ Kfu]
            #   = inv(Luu).T @ inv[I + inv(Luu)@ Kuf @ inv(D)@ Kfu @ inv(Luu).T] @ inv(Luu)
            #   = inv(Luu).T @ inv[I + W @ inv(D) @ W.T] @ inv(Luu)
            #   = inv(Luu).T @ inv(K) @ inv(Luu)
            #   = inv(Luu).T @ inv(L).T @ inv(L) @ inv(Luu)
            # loc = Ksu @ S @ Kuf @ inv(D) @ y = Ws.T @ inv(L).T @ inv(L) @ W @ inv(D) @ y
            # cov = Kss - Ksu @ inv(Kuu) @ Kus + Ksu @ S @ Kus
            #     = kss - Ksu @ inv(Kuu) @ Kus + Ws.T @ inv(L).T @ inv(L) @ Ws

            X = self.transform_inputs(train_inputs[0])
            Xnew = self.transform_inputs(inputs[0])
            Xu = self.inducing_points()
            N = X.size(-2)
            M = Xu.size(-2)

            ####### part to be cached #######
            batch_size = torch.Size([]) if self.raw_noise.ndim == 1 else torch.Size([self.raw_noise.shape[0]])
            Kuu = self.covar_module(Xu).evaluate()
            Kuu.view(*batch_size,-1)[...,::M+1] += 1e-6 # adding a small jitter term for stability
            Luu = torch.linalg.cholesky(Kuu)

            Kuf = self.covar_module(Xu,X).evaluate()

            W = torch.triangular_solve(Kuf, Luu,upper=False)[0]

            Qffdiag = W.pow(2).sum(dim=-2)
            if self.approx == 'FITC':
                Kffdiag = self.covar_module(X,diag=True)
                D = self.raw_noise_constraint.transform(self.raw_noise) + Kffdiag-Qffdiag
            else:
                D = self.raw_noise_constraint.transform(self.raw_noise) + torch.zeros_like(Qffdiag)

            W_Dinv = W/D.unsqueeze(-2)
            K = W_Dinv.matmul(W.transpose(-2,-1)).contiguous()
            K.view(*batch_size,-1)[...,:: M + 1] += 1  # add identity matrix to K
            L = torch.linalg.cholesky(K)

            # get y_residual and convert it into 2D tensor for packing
            y_residual = self.train_targets - self.mean_module(X)
            #y_2D = y_residual.reshape(-1, N).t()
            W_Dinv_y = W_Dinv.matmul(y_residual.unsqueeze(-1))
            ####### end caching part #######

            Kus = self.covar_module(Xu,Xnew).evaluate()
            Ws = torch.triangular_solve(Kus, Luu,upper=False)[0]
            pack = torch.cat((W_Dinv_y, Ws), dim=-1)
            Linv_pack = torch.triangular_solve(pack, L,upper=False)[0]
            # unpack
            Linv_W_Dinv_y = Linv_pack[...,: W_Dinv_y.shape[-1]]
            Linv_Ws = Linv_pack[..., W_Dinv_y.shape[-1] :]

            pred_mean = self.mean_module(Xnew)+Linv_W_Dinv_y.transpose(-2,-1).matmul(Linv_Ws).squeeze(-2)

            Kssdiag = self.covar_module(Xnew, diag=True)
            Qssdiag = Ws.pow(2).sum(dim=-2)
            pred_var = Kssdiag - Qssdiag + Linv_Ws.pow(2).sum(dim=-2)            

            return pred_mean,pred_var
        
    def predict(
        self,x:torch.Tensor,return_std:bool=False
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

        out_mean = self.y_mean + self.y_std*output[0]
        
        # standard deviation may not always be needed
        if return_std:
            output[1][output[1]<1e-6] = 1e-6
            out_std = output[1].sqrt()*self.y_std
            return out_mean,out_std

        return out_mean
    
    def reset_parameters(self):
        """Reset parameters by sampling from prior
        """
        for _,module,prior,closure,setting_closure in self.named_priors():
            if not closure(module).requires_grad:
                continue
            setting_closure(module,prior.expand(closure(module).shape).sample())
        
        # reset inducing-points
        with torch.no_grad():
            self.initialize(**{
                'raw_quant_inducing':torch.randn_like(self.raw_quant_inducing)
            })

            self.initialize(**{
                ('raw_thetas%d'%k):torch.randn_like(getattr(self, 'raw_thetas%d'%k)) for k in range(len(self.qual_index))
            })
