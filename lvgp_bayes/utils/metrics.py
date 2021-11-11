import numpy as np
import torch
from torch.distributions import Normal

def rrmse(y_true,y_pred):
    return np.sqrt(((y_true-y_pred)**2).mean())/y_true.std()

def nll(y_true,means,variances):
    return -Normal(means,variances.sqrt()).log_prob(y_true).mean()

def mean_interval_score(y_true,lq,uq,alpha=0.05):
    term1 = uq-lq
    term2 = 2/alpha*torch.nn.functional.relu(lq-y_true)
    term3 = 2/alpha*torch.nn.functional.relu(-uq+y_true)
    return (term1+term2+term3).mean()

def coverage(y_true,lq,uq):
    return (1.*(y_true>=lq)*(y_true<=uq)).float().mean()

def gaussian_mean_confidence_interval(means,stds,alpha=0.05):
    lq = []
    uq = []
    for i in range(means.shape[1]):
        I = torch.randint(size=(10000,),high=means.shape[0])
        sampled_means = means[I,i]
        sampled_stds = stds[I,i]
        samples = sampled_means + sampled_stds*torch.randn_like(sampled_means)
        lq.append(np.quantile(samples,alpha/2))
        uq.append(np.quantile(samples,1-alpha/2))
        
    return torch.tensor(lq),torch.tensor(uq)