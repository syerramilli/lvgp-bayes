Terminal command to run exact MAP inference

```bash
python run_script_map.py \
--save_dir runs \
--response Shear \
--num_samples 1000 \
--n_jobs 1 \ 
--n_repeats 25
```

`n_jobs` is the number of cores available to run the replicates in parallel. 

Terminal command to run MAP and fully Bayesian inference for the Sparse LVGP model with FITC approximation

```bash
python run_script.py \
--save_dir runs \
--response Shear \
--approx FITC \
--num_samples 1000 \
--n_jobs 1 \ 
--n_repeats 25
```

To estimate Sparse LVGP models with VFE approximations, change `--approx FITC` to `--approx VFE`.


Source for the data:

Emery, A., Wolverton, C. High-throughput DFT calculations of formation energy, stability and oxygen vacancy formation energy of ABO3 perovskites. Sci Data 4, 170153 (2017). https://doi.org/10.1038/sdata.2017.153