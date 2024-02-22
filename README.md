## Graph neural network application for accurate ancestry inference from Identity by Descent Graphs

Supplementary code for the paper submitted to the ICLR2024 Workshop MLGenX

As the datasets with customer data can not be made publicly available, we present the pipline that duplicates all the inference stages but is preceded by simulation stage based on significant parameters of original datasets.

### 1. Simulate 
1. Collect the significant parameters of original datasets (for reference only, can not be executed without original datasets): `python3 -m ancinf getparams dataset_folder --override_popsizes=500` produces `paramfile.json`
2. Simulate new IBD graphs with distribution of edges and edge weights with corresponding parameters: `python3 -m ancinf simulate data/paramfile.json` will produce csv files with datasets for parameters listed in `paramfile.json`

### 2. Dataset preprocessing
1. Generate dataset splits for cross-validation: `python3 -m ancinf split dataset_folder --val=20 --test=20 --count=10` will produce `datasetname_splits.json` for every `datasetname.csv` file in `dataset_folder` with 10 random splits 60:20:20

### 3. Classification 
1. Compute metrics for basic heuristics: `python3 -m ancinf heuristic dataset.csv dataset_splits.json`
2. Compute metrics for MLP network: `python3 -m ancinf mlp dataset.csv dataset_splits.json`
3. Compute metrics for selected graph neural networks: `python3 -m ancinf gnn dataset.csv dataset_splits.json`
