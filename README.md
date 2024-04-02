## Graph neural network application for accurate ancestry inference from Identity by Descent Graphs

Supplementary code 

## Stage 1. Collect dataset parameters (requires original datasets)
Input: folder with datasets, file with project description

Output: file with simulation parameters and other neccessary data for simulations

Command line:
```
Usage: python -m ancinf getparams [OPTIONS] DATADIR WORKDIR

  Collect parameters of csv files in the DATADIR listed in project file from
  WORKDIR

Options:
  --infile TEXT   Project file, defaults to project.ancinf
  --outfile TEXT  Output file with simulation parameters, defaults to project
                  file with '.params' extension
```

Python import: 
```
from ancinf.utils.simulate import collectparams
params = collectparams(datadir, workdir, "project.ancinf")
```


## Stage 1'. Preprocess original dataset (requires original datasets)
Input: folder with datasets, file with project description

Output: filtered datasets, train-validate-test splits, file with a list of experiments

Command line:
```
Usage: python -m ancinf preprocess [OPTIONS] DATADIR WORKDIR

  Filter datsets from DATADIR, generate train-val-test splits and experiment
  list file in WORKDIR

Options:
  --infile TEXT   Project file, defaults to project.ancinf
  --outfile TEXT  Output file with experiment list, defaults to project file
                  with '.explist' extension
  --seed INTEGER  Random seed.
```

Python import: 
```
from ancinf.utils.simulate import preprocess
preprocess(datadir, workdir, "nosim.ancinf", "nosim.explist", rng)
```


## Stage 2. Simulate 
Input: file with simulation parameters

Output: simulated datasets, train-validate-test splits, file with a list of experiments

Command line:
```
Usage: python -m ancinf simulate [OPTIONS] WORKDIR

  Generate ibd graphs, corresponding slpits and experiment list file for
  parameters in INFILE

Options:
  --infile TEXT   File with simulation parameters, defaults to project.params
  --outfile TEXT  Output file with experiment list, defaults to project file
                  with '.explist' extension
  --seed INTEGER  Random seed.

```

Python import: 
```
from ancinf.utils.simulate import simulateandsave
simulateandsave(workdir, "smallproject.params", "smallproject.explist", rng)
```


## Cross-validation stages

## Stage 3 
1. Compute metrics for basic heuristics: `python3 -m ancinf heuristic dataset.csv dataset_splits.json`

## Stage 4 
2. Train MLP network and compute its metrics: `python3 -m ancinf mlp dataset.csv dataset_splits.json`

## Stage 5 
3. Train selected graph neural networks and compute their metrics: `python3 -m ancinf gnn dataset.csv dataset_splits.json`

## Inference stage
