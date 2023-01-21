import os
import time
import torch
import random
import argparse
import numpy as np
import pandas as pd
from joblib import dump,Parallel,delayed,load

from scipy.io import loadmat
from lvgp_bayes.models import LVGPR
from lvgp_bayes.optim import fit_model_scipy, run_hmc_numpyro
from lvgp_bayes.utils.variables import CategoricalVariable
from lvgp_bayes.utils.input_space import InputSpace
from lvgp_bayes.utils.metrics import rrmse,mean_interval_score,coverage
from lvgp_bayes.utils.metrics import gaussian_mean_confidence_interval

# for MCMC
import jax
from numpyro.diagnostics import summary
jax.config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser('Spinels MAP vs fully Bayesian')
parser.add_argument('--save_dir',type=str,required=True)
parser.add_argument('--response',type=str,required=True)
parser.add_argument('--estimation',type=str,required=True,choices=['map','mle','mcmc'])
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
res = loadmat('data_spinels.mat')
properties = res['Properties']
all_combs = torch.tensor(res['compounds_coded']).double()-1

mats = pd.DataFrame(
    res['compounds_coded']-1,
    columns = ['$A$','$M1$','$M2$','$Q$']
)
mats['stability'] = properties[:,0]
mats['bandgap'] = properties[:,1]

config = InputSpace()
cols = ['$A$','$M1$','$M2$','$Q$']
num_levels_list = [3,6,5,3]
config.add_inputs([
    CategoricalVariable(name, np.arange(num_levels)) \
        for name,num_levels in zip(cols,num_levels_list)
])

all_responses = torch.from_numpy(mats[args.response].values).double()
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

    train_x,test_x = all_combs[train_idxs,:],all_combs[test_idxs,:]
    train_y,test_y = all_responses[train_idxs],all_responses[test_idxs]

    # save training and test indices
    dump(np.array(train_idxs),os.path.join(save_dir_seed,'train_idxs.pkl'))
    dump(test_idxs,os.path.join(save_dir_seed,'test_idxs.pkl'))

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