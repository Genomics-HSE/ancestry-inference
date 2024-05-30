## Graph neural network application for accurate ancestry inference from Identity by Descent Graphs

Supplementary code 

The pipeline below can be used in following ways:
1. IBD dataset + .ancinf file ---(stage1)---> .params file ---(stage2)---> simulated datasets + .explist file ---(stage3)---> .results file
2. IBD dataset + .ancinf file ---(stage1')---> filtered dataset + .explist file ---(stage3)---> .results file
3. Manually composed .params file ---(stage2)---> simulated datasets + .explist file ---(stage3)---> .results file

After stages 1 or 1' the project work directory becomes independent of the data directory, as it now contains either filtered original datasets or computed parameters for further simulations.

After stage 3 the best weights for selected neural networks are kept, so they can later be used for inference.

4. Training dataset + trained model weights + dataset with edges from nodes with unknown label to training dataset ---(stage4)---> .inferred file

# I. Preprocessing and simulation

## Stage 1. Collect dataset parameters (requires original datasets)
Input: folder with datasets, file with project description

Output: file with simulation parameters and other neccessary data for simulations

Command line:
```
Usage: python3 -m ancinf getparams [OPTIONS] DATADIR WORKDIR

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
Usage: python3 -m ancinf preprocess [OPTIONS] DATADIR WORKDIR

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
Usage: python3 -m ancinf simulate [OPTIONS] WORKDIR

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

## Stage 3. Compute metrics for selected classifiers: 
Input: Original or simulated datasets, train-validate-test splits, file with a list of experiments

Output: f1 macro srores for selected classifiers

Command line:
```
Usage: python -m ancinf crossval [OPTIONS] WORKDIR

  Run crossvalidation for classifiers including heuristics, community
  detections, GNNs and MLP networks

Options:
  --infile TEXT        File with experiment list, defaults to project.explist
  --outfile TEXT       File with classification metrics, defaults to project
                       file with '.result' extension
  --seed INTEGER       Random seed
  --processes INTEGER  Number of parallel workers
  --fromexp INTEGER    The first experiment to run
  --toexp INTEGER      Last experiment (not included)
  --fromsplit INTEGER  The first split to run
  --tosplit INTEGER    Last split (not included)
  --gpu INTEGER        GPU
  --gpucount INTEGER   GPU count
```


# III. Inference 

## Stage 4. Inference 

Input: datasets for train and inference, model weights

Output: inference results

```
Usage: python -m ancinf infer [OPTIONS] WORKDIR TRAINDF INFERDF MODEL WEIGHTS

  traindf: Dataset on which the model was trained

  inferdf, Dataset with nodes with classes to be inferred (labelled 'unknown')

  model: Model name

  weights: Weights file
```

# IV. Utilities
## Combine .result files from a folder


# Project file format

.ancinf file has the following sections:

1. "datasets" required for stages 1 and 1'. Keys correspond to csv file names (without .csv extension) in DATADIR, values contain filters to be applied to corresponding datasets.
2. "simulator" section required for stage 2:
 - "type": "exponential" the only underlying edge weight distribution is exponential
 - "offset": shift of the exponential probability density function
3. "experiments" section required for stage 2 contains lists of factors for different simulation parameters. Parameters from .params file are multiplied by every combination of the following factors, and each combination is considered to be "an experiment" in .explist flle. If factors are not needed, keep the lists with one factor only: [1.0], meaning only one experiment per parameter set in .params file will be performed. In case of stage 1' there is also only one "experiment", that is, the filtered original dataset.
  - "population_scale": multiply every population size 
  - "diag_edge_probability_scale": multiply edge probability inside every population (diagonal elements of edge probability matrix)
  - "nondg_edge_probability_scale": multiply edge probability between every pair of populations (out-of-diagonal elements of edge probability matrix)
  - "diag_weight_scale": multiply average weight of edge inside every population (diagonal element of average weight matrix)
  - "nondg_weight_scale": multiply average weight of edge between every pair of populations (out-of-diagonal elements of average weight matrix)
  - "all_edge_probability_scale":  multiply all elements of edge probability matrix
  - "all_weight_scale": multiply all elements of average weight matrix
4. "crossvalidation" section required for stage 3:
 - "cleanshare": share of every population to be excluded from train-val-test loop completely. If cleanshare is specified, then special csv datafile (ending with "clean") containing these excluded nodes will be created in the work directory and for every trained network an inference will be performed by adding nodes from this file one by one, and finally f1 macro of this inference will be computed and stored in results with "clean" prefix.
 - "maskshare": share of every population to be marked "unknown"
 - "valshare" and "testshare": share of every population to include into validation and test datasets respectively. If "cleanshare" was specified, then these shares are taken from nodes that are left (non excluded).
 - "split_count": number of random splits into train-val-test subsets for crossvalidation.
 - "log_weights": False means that edge weights in the training graphs are taken as they are in the datafile, True sets logarithms of these weights.
 - "heuristics": list of heuristic classifiers to be used in cross-validation. Possible values: ["EdgeCount", "EdgeCountPerClassize", "SegmentCount", "LongestIbd", "IbdSum", "IbdSumPerEdge"]
 - "community_detection": list of community detection algorithms to be used in cross-validation. Possible values: ["Spectral", "Agglomerative", "Girvan-Newmann", "LabelPropagation", "RelationalNeighbor"].
 - "mlps": list of multilayer perceptron architecture NN classifiers to be used in cross-validation. Possible values: ["MLP_3l_128h", "MLP_3l_512h", "MLP_9l_128h", "MLP_9l_512h"]
 - "gnns": list of graph neural network classifiers to be used in cross-validation. Possible values: ["TAGConv_9l_512h_nw_k3", 
                    "TAGConv_9l_128h_k3",                    
                    "TAGConv_3l_128h_w_k3",
                    "TAGConv_3l_512h_w_k3",
                    "TAGConv_9l_512h_nw_k3_gb",
                    "TAGConv_9l_128h_k3_gb",                    
                    "TAGConv_3l_128h_w_k3_gb",
                    "TAGConv_3l_512h_w_k3_gb",
                    "GCNConv_3l_128h_w",
                    "GCNConv_3l_128h_w_gb",
                    "GINNet",
                    "GINNet_narrow_short", 
                    "GINNet_wide_short", 
                    "GINNet_narrow_long", 
                    "GINNet_wide_long",
                    "GINNet_gb",
                    "GINNet_narrow_short_gb", 
                    "GINNet_wide_short_gb", 
                    "GINNet_narrow_long_gb", 
                    "GINNet_wide_long_gb",
                    "AttnGCN",
                    "AttnGCN_narrow_short", 
                    "AttnGCN_wide_short", 
                    "AttnGCN_narrow_long", 
                    "AttnGCN_wide_long",
                    "AttnGCN_gb",
                    "AttnGCN_narrow_short_gb", 
                    "AttnGCN_wide_short_gb", 
                    "AttnGCN_narrow_long_gb", 
                    "AttnGCN_wide_long_gb"]

5. To pass non-default neural net parameters or training parameters it is possible to change "gnns" or "mlps" list into dictionary with keys from the list above and values of dict with corresponding parameter names and values.
