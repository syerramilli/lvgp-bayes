Sample terminal command to run MCMC inference for the LVGP model of the Shear modulus as a function of the composition across all 25 replicates of training-test splits used in the paper.

```bash
python run_script.py \
--save_dir runs \
--model lvgp \
--estimation mcmc \
--response Shear \
--num_samples 100 \
--n_jobs 1 \ 
--n_repeats 25
```

`n_jobs` is the number of cores available to run the replicates in parallel. 

Source for the data:

Balachandran, P., Xue, D., Theiler, J. et al. Adaptive Strategies for Materials Design using Uncertainties. Sci Rep 6, 19660 (2016). https://doi.org/10.1038/srep19660