import os
import time
import torch
import random
import argparse
import numpy as np
import pandas as pd
from joblib import dump,Parallel,delayed,load

from lvgp_bayes.models.sparselvgp import SparseLVGPR
from lvgp_bayes.optim.fitc_scipy import fit_fitc_model_scipy
from lvgp_bayes.optim.numpyro_hmc import run_hmc_numpyro
from lvgp_bayes.utils.variables import CategoricalVariable
from lvgp_bayes.utils.input_space import InputSpace
from lvgp_bayes.utils.metrics import rrmse,mean_interval_score,coverage
from lvgp_bayes.utils.metrics import gaussian_mean_confidence_interval

from copy import deepcopy
from typing import Dict

import jax
from numpyro.diagnostics import summary
jax.config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser('ABO3 MAP vs fully Bayesian')
parser.add_argument('--save_dir',type=str,required=True)
parser.add_argument('--response',type=str,required=True)
parser.add_argument('--approx',type=str,required=True)
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
dat = pd.read_csv('ABO3perovskites.csv',na_values='-')
dat = dat[~dat['Formation energy [eV/atom]'].isna()]

cols = ['A','B']
levels_list = [
    sorted(dat[col].unique().tolist()) for col in cols
]

config = InputSpace()
config.add_inputs([
    CategoricalVariable(col,levels=levels) \
    for col,levels in zip(cols,levels_list)
])

# encode categorical variables as integers
# and scale numerical variables to the unit hypercube
all_x = torch.from_numpy(np.array([
    config.get_array_from_dict(row.to_dict())\
    for _,row in dat[config.get_variable_names()].iterrows()
]))

if args.response == 'formation':
    target = "Formation energy [eV/atom]"
elif args.response == 'stability':
    target = "Stability [eV/atom]"
else:
    raise RuntimeError

# response - formation energy
all_y = torch.from_numpy(dat[target].values)
#%%

def set_seed(seed:int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def main_script(seed):
    jax.config.update("jax_enable_x64", True)
    save_dir_seed = os.path.join(save_dir,'seed_%d'%seed)
    if not os.path.exists(save_dir_seed):
        os.makedirs(save_dir_seed)

    rng = np.random.RandomState(seed)
    train_idxs = rng.choice(all_x.shape[0],args.num_samples,replace=False)
    test_idxs = np.array([idx for idx in np.arange(all_x.shape[0]) if idx not in train_idxs])

    train_x,test_x = all_x[train_idxs,:],all_x[test_idxs,:]
    train_y,test_y = all_y[train_idxs],all_y[test_idxs]

    # save training and test indices
    dump(np.array(train_idxs),os.path.join(save_dir_seed,'train_idxs.pkl'))
    dump(test_idxs,os.path.join(save_dir_seed,'test_idxs.pkl'))

    set_seed(seed)
    model = SparseLVGPR(
        train_x=train_x,
        train_y=train_y,
        quant_correlation_class='RBFKernel',
        qual_index=config.qual_index,
        quant_index=config.quant_index,
        num_levels_per_var=list(config.num_levels.values()),
        lv_dim=2,
        num_inducing=50,
        approx=args.approx,
        lb_noise=1e-4
    ).double()
    #model.likelihood.register_prior('noise_prior',ExpHalfHorseshoePrior(0.1,lb=1e-6),'raw_noise')

    start_time = time.time()
    if args.approx == 'FITC':
        _ = fit_fitc_model_scipy(model,num_restarts=9,options={'ftol':1e-6,'maxfun':1000})
    else:
        # VFE model is difficult to optimize
        # initialize from the FITC solution
        model_fitc = SparseLVGPR(
            train_x=train_x,
            train_y=train_y,
            quant_correlation_class='RBFKernel',
            qual_index=config.qual_index,
            quant_index=config.quant_index,
            num_levels_per_var=list(config.num_levels.values()),
            lv_dim=2,
            num_inducing=50,
            approx='FITC',
            lb_noise=1e-4
        ).double()
        _ = fit_fitc_model_scipy(model_fitc,num_restarts=9,options={'ftol':1e-6,'maxfun':1000})

        model.load_state_dict(deepcopy(model_fitc.state_dict()))
        _ = fit_fitc_model_scipy(model,num_restarts=0,options={'ftol':1e-6,'maxfun':1000})

    fit_time_map = time.time() - start_time

    # save MAP state
    torch.save(model.state_dict(),os.path.join(save_dir_seed,'state_%s_map.pth'%args.approx))

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
    dump(stats_map, os.path.join(save_dir_seed,'stats_%s_map.pkl'%args.approx))
    
    # # load MAP state
    # state_dict = torch.load(os.path.join(save_dir_seed,'state_fitc_map.pth'))
    # model.load_state_dict(state_dict)

    # fit_time_map = load(os.path.join(save_dir_seed,'stats_fitc_map.pkl'))['training_time']

    # run mcmc
    set_seed(seed)
    model.train()
    start_time = time.time()
    mcmc_runs = run_hmc_numpyro(
        model,
        num_samples=1500,warmup_steps=1500,
        max_tree_depth=7,
        disable_progbar=True,
        num_chains=1,
        num_model_samples=100,
        seed=seed,
        initialize_from_state=True
    )
    fit_time_mcmc = time.time()-start_time + fit_time_map
    torch.save(model.state_dict(),os.path.join(save_dir_seed,'state_%s_mcmc.pth'%args.approx))
    
    diagnostics = summary(mcmc_runs.get_samples(),group_by_chain=False)
    dump(diagnostics,os.path.join(save_dir_seed,'diagnostics_%s.pkl'%args.approx))
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
    dump(stats_mcmc, os.path.join(save_dir_seed,'stats_%s_mcmc.pkl'%args.approx))

#%%
seeds = np.linspace(100,1000,args.n_repeats).astype(int)

Parallel(n_jobs=args.n_jobs,verbose=0)(
    delayed(main_script)(seed) for seed in seeds
)