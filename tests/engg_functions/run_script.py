import os
import time
import torch
import random
import argparse
import numpy as np
from joblib import dump,Parallel,delayed,load

import gpytorch
import configs
import functions

from lvgp_bayes.models import LVGPR
from lvgp_bayes.optim import run_hmc_numpyro,fit_model_scipy

from lvgp_bayes.utils.metrics import rrmse,mean_interval_score,coverage
from lvgp_bayes.utils.metrics import gaussian_mean_confidence_interval

# for MCMC
import jax
from numpyro.diagnostics import summary
jax.config.update("jax_enable_x64", True)

from copy import deepcopy
from typing import Dict

parser = argparse.ArgumentParser('Engg functions MAP vs fully Bayesian')
parser.add_argument('--save_dir',type=str,required=True)
parser.add_argument('--which_func',type=str,required=True)
parser.add_argument('--estimation',type=str,required=True)
parser.add_argument('--train_factor',type=int,required=True)
parser.add_argument('--n_jobs',type=int,required=True)
parser.add_argument('--n_repeats',type=int,default=25)
args = parser.parse_args()

save_dir = os.path.join(
    args.save_dir,
    '%s/train_factor_%d'%(args.which_func,args.train_factor),
)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#%% configuration object and function definition
config,V,maps = getattr(configs,args.which_func)()

obj = getattr(functions,args.which_func)

def obj_new(params)-> float:
    new_params = deepcopy(params)
    new_params.update(maps[new_params['t']])
    return obj(new_params)

def set_seed(seed:int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

#%% test data
test_x = torch.from_numpy(config.random_sample(np.random.RandomState(456),1000))
test_y = [None]*test_x.shape[0]

for i,x in enumerate(test_x):
    test_y[i] = obj_new(config.get_dict_from_array(x.numpy()))
    
# create tensor objects
test_y = torch.tensor(test_y).to(test_x)

#%%

def main_script(seed):
    save_dir_seed = os.path.join(save_dir,'seed_%d'%seed)
    if not os.path.exists(save_dir_seed):
        os.makedirs(save_dir_seed)

    n_cat = V.shape[0]
    n_train = args.train_factor*n_cat
    rng = np.random.RandomState(seed)
    train_x = torch.from_numpy(config.latinhypercube_sample(rng,n_train))
    train_y = [None]*n_train

    for i,x in enumerate(train_x):
        train_y[i] = obj_new(config.get_dict_from_array(x.numpy()))
        
    # create tensor objects
    train_y = torch.tensor(train_y).to(train_x)

    # save training data
    torch.save(train_x,os.path.join(save_dir_seed,'train_x.pt'))
    torch.save(train_y,os.path.join(save_dir_seed,'train_y.pt'))

    set_seed(seed)
    model = LVGPR(
        train_x=train_x,
        train_y=train_y,
        quant_correlation_class='RBFKernel',
        qual_index=config.qual_index,
        quant_index=config.quant_index,
        num_levels_per_var=list(config.num_levels.values()),
        lv_dim=2,
    ).double()

    if args.estimation == 'mle':
        # fix precision hyperparameters to 1 and disable gradients
        for layer in model.lv_mapping_layers:
            layer.raw_precision.requires_grad_(False)
            layer.initialize(**{'precision':torch.ones(1).double()})

    if args.estimation in ['mle','map']:
        start_time = time.time()
        _ = fit_model_scipy(
            model,
            add_prior= True if args.estimation == 'map' else False,
            num_restarts=15,options={'ftol':1e-6,'maxfun':1000}
        )
        fit_time = time.time() - start_time

        # generate predictions
        with torch.no_grad():
            # set return_std = False if standard deviation is not needed 
            test_pred0,test_pred_std0 = model.predict(test_x,return_std=True)

        # print RRMSE
        lq,uq = test_pred0-1.96*test_pred_std0,test_pred0+1.96*test_pred_std0
        stats = {
            'rrmse':rrmse(test_y,test_pred0).item(),
            'mis':mean_interval_score(test_y,lq,uq,0.05).item(),
            'coverage':coverage(test_y,lq,uq).item(),
            'training_time':fit_time
        }
    else:
        start_time = time.time()
        mcmc_runs = run_hmc_numpyro(
            model,
            num_samples=1500,warmup_steps=1500,
            max_tree_depth=7,
            disable_progbar=True,
            num_chains=1,
            num_model_samples=100,
            seed=seed
        )
        fit_time = time.time()-start_time

        diagnostics = summary(mcmc_runs.get_samples(),group_by_chain=False)
        dump(diagnostics,os.path.join(save_dir_seed,'mcmc_diagnostics.pkl'))

        # predictions
        with torch.no_grad():
            means,stds = model.predict(test_x,return_std=True)
        
        lq,uq = gaussian_mean_confidence_interval(means,stds)
        stats = {
            'rrmse':rrmse(test_y,means.mean(axis=0)).item(),
            'mis':mean_interval_score(test_y,lq,uq,0.05).item(),
            'coverage':coverage(test_y,lq,uq).item(),
            'training_time':fit_time
        }
    
    # save state and stats
    torch.save(model.state_dict(),os.path.join(save_dir_seed,'%s_state.pth'%(args.estimation,)))
    dump(stats, os.path.join(save_dir_seed,'stats_%s.pkl'%(args.estimation,)))

#%%
seeds = np.linspace(100,1000,args.n_repeats).astype(int)

Parallel(n_jobs=args.n_jobs,verbose=0)(
    delayed(main_script)(seed) for seed in seeds
)