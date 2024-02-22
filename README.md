## Graph neural network application for accurate ancestry inference from Identity by Descent Graphs

Supplementary code for the paper submitted to the ICLR2024 Workshop MLGenX

As the datasets with customer data can not be made publicly available, we present the pipline that duplicates all the inference stages but is preceded by simulation stage based on significant parameters of original datasets.

### 1. Simulate 
1. Collect the significant parameters of original datasets (for reference only, can not be executed without original datasets): `python3 pipeline.py getparams dataset_folder` produces `paramfile.txt`
2. Simulate new IBD graphs with distribution of edges and edge weights with corresponding parameters: `python3 pipeline.py simulate data/paramfile.txt` will produce csv files with datasets for parameters listed in `paramfile.txt`

### 2. Dataset preprocessing
1. Generate dataset splits for cross-validation: `python3 pipeline.py split dataset.csv 10` will produce `dataset_splits.json` with 10 random splits

### 3. Classification 
1. Compute metrics for basic heuristics: `python3 pipeline.py heuristic dataset.csv dataset_splits.json`
2. Compute metrics for MLP network: `python3 pipeline.py mlp dataset.csv dataset_splits.json`
3. Compute metrics for selected graph neural networks: `python3 pipeline.py gnn dataset.csv dataset_splits.json`
