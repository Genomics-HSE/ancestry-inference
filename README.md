## Graph neural network application for accurate ancestry inference from Identity by Descent Graphs

Supplementary code 

# I. Preprocessing and simulation

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


# II. Cross-validation 

## Stage 3. Compute metrics for basic heuristics: 
Input: Original or simulated datasets, train-validate-test splits, file with a list of experiments

Output: heuristic classification metrics

Command line:
```
Usage: python -m ancinf heuristics [OPTIONS] WORKDIR

  Run heuristics

Options:
  --infile TEXT   File with experiment list, defaults to project.explist
  --outfile TEXT  File with classification metrics, defaults to project 
                  file with '.result' extension
  --seed INTEGER  Random seed
```

Python import: 
```
from ancinf.utils.runheuristic import runheuristics
runheuristics(workdir, 'project.explist', rng)
```


## Stage 4 
2. Train MLP network and compute its metrics: `python3 -m ancinf mlp dataset.csv dataset_splits.json`

## Stage 5. Train selected graph neural networks and compute their metrics: 

Input: Original or simulated datasets, train-validate-test splits, file with a list of experiments

Output: GNN classification metrics

Command line:
```
Usage: python -m ancinf gnn [OPTIONS] WORKDIR

  Run heuristics

Options:
  --infile TEXT   File with experiment list, defaults to project.explist
  --outfile TEXT  File with classification metrics, defaults to project 
                  file with '.result' extension
  --seed INTEGER  Random seed
```

Python import: 
```
from ancinf.utils.runheuristic import rungnn
rungnn(workdir, 'project.explist', rng)
```




# III. Inference 
