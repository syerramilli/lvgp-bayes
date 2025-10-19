import os
import re
import joblib
import numpy as np
import pandas as pd
from itertools import product

def diffboxplot(data,x,y,ax,color,positions=None,transform=None):
    seqs = []
    labels = []
    for group,dat_group in data.groupby(x):
        labels.append(group)
        seq = dat_group[y].dropna().values
        if transform is not None:
            seq = transform(seq)
        seqs.append(seq)

    bp= ax.boxplot(
        x=seqs,labels=labels,
        positions=positions,
        medianprops={'color':color},
        boxprops={'color':color},
        whiskerprops={'color':color},
        capprops={'color':color},
        flierprops={
            'marker':'.',
            'markerfacecolor':color,
            'markeredgecolor':color
        },
        patch_artist=True
    )
    
    # set background to white
    for box in bp['boxes']:
        box.set_facecolor('w')
    return bp

def extract_data(dir_folder,num_levels):
    datlist = []
    for subfolder in os.listdir(dir_folder):
        subfolder_path = os.path.join(dir_folder,subfolder)
        train_factor = int(re.findall(r'[1-9]',subfolder)[0])

#         file_identifiers = list(product(['','lvgpdesc_'],['map','mcmc']))
#         est_identifiers = list(product(['',' with descs.'],['MAP','MCMC']))
        identifiers = ['map','mcmc']
        datlist_sub = []
        for f in identifiers:
            try:
                tmp =  pd.DataFrame([
                    joblib.load(os.path.join(subfolder_path,seed,'stats_%s.pkl'%f)) \
                        for seed in sorted(os.listdir(subfolder_path))
                ])
                tmp['estimation'] = f.upper()
                datlist_sub.append(tmp)
            except:
                continue
        
        tmp = pd.concat(datlist_sub,axis=0,ignore_index=True)
        tmp['ntrain'] = train_factor*num_levels
        datlist.append(tmp)
    return pd.concat(datlist,axis=0,ignore_index=True)