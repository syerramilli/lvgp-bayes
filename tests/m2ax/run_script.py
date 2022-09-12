import os
import time
import torch
import random
import argparse
import numpy as np
import pandas as pd
from joblib import dump,Parallel,delayed,load

import gpytorch
from lvgp_bayes.models import GPR,LVGPR
from lvgp_bayes.models.orth_lvgp import OrthLVGPR
from lvgp_bayes.optim import fit_model_scipy
from lvgp_bayes.optim.hmc_jit import run_hmc_jit
from lvgp_bayes.priors import ExpHalfHorseshoePrior
from lvgp_bayes.utils.variables import CategoricalVariable
from lvgp_bayes.utils.input_space import InputSpace
from lvgp_bayes.utils.metrics import rrmse,mean_interval_score,coverage
from lvgp_bayes.utils.metrics import gaussian_mean_confidence_interval

from copy import deepcopy
from typing import Dict

parser = argparse.ArgumentParser('M2AX MAP vs fully Bayesian')
parser.add_argument('--save_dir',type=str,required=True)
parser.add_argument('--model',type=str,required=True,choices=['gp','new_lvgp','orth_lvgp'])
parser.add_argument('--response',type=str,required=True)
parser.add_argument('--num_samples',type=int,required=True)
parser.add_argument('--n_jobs',type=int,required=True)
parser.add_argument('--n_repeats',type=int,default=25)
args = parser.parse_args()

save_dir = os.path.join(
    args.save_dir,
    args.response,'%d-samples'%args.num_samples
)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#%%
dat = pd.read_excel('M2AX_data.xls')
config = InputSpace()
col_names = ['%s-site element'%l for l in ['M','A','X']]
elems = [
    CategoricalVariable(name=name,levels=dat[name].unique()) \
    for name in col_names
]
config.add_inputs(elems)

all_combs = torch.from_numpy(np.array([
    config.get_array_from_dict(row) for _,row in dat[config.get_variable_names()].iterrows()
])).double()


if args.response == 'Young':
    target = "E (Young's modulus)"
elif args.response == 'Shear':
    target = "G (Shear modulus)"
elif args.response == 'Bulk':
    target = "B (Bulk modulus)"

all_responses = -torch.tensor(dat[target]).double()

#%% descriptors
if args.model == 'gp':
    descs_dat = [
        (
            dat[[col for col in dat.columns if site in col]]
            .drop_duplicates()
            .reset_index(drop=True)
            .set_index('%s-site element'%site)
            .apply(lambda x: (x-x.min())/(x.max()-x.min()))
        ) for site in ['M','A']
    ]

    all_descs = [
        torch.from_numpy(descs_dat[i].values)[
            all_combs[:,i].long(),:
        ] for i in config.qual_index
    ]
    all_descs.append(all_combs[:,config.quant_index])
    all_descs = torch.cat(all_descs,dim=1).double()

#%%

def set_seed(seed:int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def main_script(seed):
    save_dir_seed = os.path.join(save_dir,'seed_%d'%seed)
    if not os.path.exists(save_dir_seed):
        os.makedirs(save_dir_seed)

    rng = np.random.RandomState(seed)
    train_idxs = rng.choice(all_combs.shape[0],args.num_samples,replace=False)
    test_idxs = np.array([idx for idx in np.arange(all_combs.shape[0]) if idx not in train_idxs])
    train_x,test_x = None,None
    if args.model == 'gp':
        train_x,test_x = all_descs[train_idxs,:],all_descs[test_idxs,:]
    else:
        train_x,test_x = all_combs[train_idxs,:],all_combs[test_idxs,:]
    
    train_y,test_y = all_responses[train_idxs],all_responses[test_idxs]

    # save training and test indices
    dump(train_idxs,os.path.join(save_dir_seed,'train_idxs.pkl'))
    dump(test_idxs,os.path.join(save_dir_seed,'test_idxs.pkl'))

    set_seed(seed)
    if args.model == 'gp':
        model = GPR(
            train_x=train_x,
            train_y=train_y,
            correlation_kernel='RBFKernel',
        ).double()
    elif args.model == 'new_lvgp':
        model = LVGPR(
            train_x=train_x,
            train_y=train_y,
            quant_correlation_class='RBFKernel',
            qual_index=config.qual_index,
            quant_index=config.quant_index,
            num_levels_per_var=list(config.num_levels.values()),
            lv_dim=2,
        ).double()
    else:
        model = OrthLVGPR(
            train_x=train_x,
            train_y=train_y,
            quant_correlation_class='RBFKernel',
            qual_index=config.qual_index,
            quant_index=config.quant_index,
            num_levels_per_var=list(config.num_levels.values()),
            lv_dim=2,
        ).double()
    
    #model.likelihood.register_prior('raw_noise_prior',ExpHalfHorseshoePrior(0.1),'raw_noise')
    start_time = time.time()
    _ = fit_model_scipy(model,num_restarts=15,options={'ftol':1e-6,'maxfun':1000})
    fit_time_map = time.time() - start_time

    # save MAP state
    torch.save(model.state_dict(),os.path.join(save_dir_seed,'map_state_%s.pth'%args.model))

    # generate predictions
    with torch.no_grad():
        # set return_std = False if standard deviation is not needed 
        test_pred0,test_pred_std0 = model.predict(test_x,return_std=True)

    # print RRMSE
    lq,uq = test_pred0-1.96*test_pred_std0,test_pred0+1.96*test_pred_std0
    stats_map = {
        'rrmse':rrmse(test_y,test_pred0).item(),
        'mis':mean_interval_score(test_y,lq,uq,0.05).item(),
        'coverage':coverage(test_y,lq,uq).item(),
        'training_time':fit_time_map
    }
    dump(stats_map, os.path.join(save_dir_seed,'stats_map_%s.pkl'%args.model))
    
    # run mcmc
    set_seed(seed)
    model.train()
    start_time = time.time()
    with gpytorch.settings.cholesky_jitter(1e-4):
        mcmc_runs = run_hmc_jit(
            model,
            warmup_steps=1000,
            num_samples=1000,
            num_model_samples=100,
            num_chains=3
        )
    fit_time_mcmc = time.time()-start_time
    torch.save(mcmc_runs[0].diagnostics(),os.path.join(save_dir_seed,'mcmc_diagnostics_%s.pth'%args.model))
    torch.save(model.state_dict(),os.path.join(save_dir_seed,'mcmc_state_%s.pth'%args.model))
    
    # predictions
    with torch.no_grad():
        means,stds = model.predict(test_x,return_std=True)
    
    lq,uq = gaussian_mean_confidence_interval(means,stds)
    stats_mcmc = {
        'rrmse':rrmse(test_y,means.mean(axis=0)).item(),
        'mis':mean_interval_score(test_y,lq,uq,0.05).item(),
        'coverage':coverage(test_y,lq,uq).item(),
        'training_time':fit_time_mcmc
    }
    dump(stats_mcmc, os.path.join(save_dir_seed,'stats_mcmc_%s.pkl'%args.model))

#%%
seeds = np.linspace(100,1000,args.n_repeats).astype(int)

Parallel(n_jobs=args.n_jobs,verbose=0)(
    delayed(main_script)(seed) for seed in seeds
)