import pandas as pd
import importlib
import genlink
import torch
import numpy as np
import random
importlib.reload(genlink)
from ..utils.genlink import DataProcessor, NullSimulator, Trainer, TAGConv_3l_128h_w_k3, TAGConv_3l_512h_w_k3

def run_inference(dataframe, splits, task_name):
    dp = DataProcessor(dataframe)

    dp.load_train_valid_test_nodes(splits['train'], splits['test'], splits['val'], 'numpy')

    dp.make_train_valid_test_datasets_with_numba('one_hot', 'homogeneous', 'multiple', 'multiple', task_name)

    trainer = Trainer(dp, TAGConv_3l_128h_w_k3, 0.0001, 5e-5, torch.nn.CrossEntropyLoss, torch.tensor([1., 1.]).to('cuda'), 10, r"C:\HSE\genotek-nationality-analysis\runs\nc_null", 2, 20)

    return trainer.run()