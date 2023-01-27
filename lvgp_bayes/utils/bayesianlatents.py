import torch
import numpy as np
from ..optim.multistart import MultiStartOptimizer
from typing import Union

class LatentEmbeddingsObj(MultiStartOptimizer):
    def __init__(
        self,latents,base_kernel,ndims=2,
        **kwargs):
        
        vec_dim = latents.ndim
        self.base_kernel = base_kernel.double()
        if self.base_kernel.has_lengthscale:
            self.base_kernel.raw_lengthscale.requires_grad_(False)
        
        
        self.corr = self.base_kernel(latents).evaluate()

        self.ndims = ndims
        self.num_latents = self.ndims*self.corr.shape[-1]-int(self.ndims*(self.ndims+1)/2)
        
        super().__init__(
            obj = self.scipy_obj,
            lb = -np.inf,
            ub = np.inf,
            jac=True,
            **kwargs
        )

    def generate_init(self):
        return self.rng.randn(self.num_starts,self.num_latents)
    
    def run(self):
        return self.embedding(super(LatentEmbeddingsObj,self).run()).numpy()

    def scipy_obj(self,x):
        emb,X = self.embedding(x,grad=True)
        corr_emb = (
            self.base_kernel(emb)
            .evaluate()
            .unsqueeze(0)
            .repeat(self.corr.shape[0],1,1)
        )
        diff = (corr_emb-self.corr).triu(1)
        loss = (0.5*(diff**2).mean()).log()
        grad=torch.autograd.grad(loss,X)[0].contiguous().view(-1).numpy()
        
        return loss.item(),grad
    
    def embedding(self,x:np.ndarray,grad:bool=False):
        X = (
            torch.from_numpy(x)
            .contiguous()
            .requires_grad_(grad)
        )
        if self.ndims==1:
            emb = torch.cat([
                torch.zeros((1,1)).to(X),
                torch.nn.functional.softplus(X[:1]).reshape(-1,1),
                X[1:].reshape(-1,1)
            ],axis=0)
        elif self.ndims==2:
            emb = torch.cat([
                torch.zeros((1,2)).to(X),
                torch.cat([
                    1e-3+torch.nn.functional.softplus(X[0:1]),torch.zeros(1).to(X)
                ],axis=-1).reshape(1,-1),
                torch.cat([
                    X[1:2],torch.nn.functional.softplus(X[2:3])+1e-3
                ],axis=-1).reshape(1,-1),
                X[3:].reshape(-1,2)
            ],axis=0)

        if grad:
            return emb,X
        else:
            return emb