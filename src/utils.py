import os
import torch
import random
import numpy as np

from importlib import import_module
from collections import defaultdict

# Text Converter for Algorithm Name
def algorithmTextConverter(algorithm):
    if algorithm == 'fedavg':
        return 'FedAvg'
    elif algorithm == 'fedhm':
        return 'FedHM'
    elif algorithm == 'fedloru':
        return 'FedLoRU'

# Range for Args
class Range:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        
    def __eq__(self, other):
        return self.start <= other <= self.end
    
    def __str__(self):
        return f'Specificed Range: [{self.start:.2f}, {self.end:.2f}]'

# Set seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f'[SEED] ...seed is set: {seed}!')


# Metric Manager
# This class is used to manage the metrics for evaluation
class MetricManager:
    def __init__(self, eval_metrics):
        self.metric_funcs = {
            name: import_module(f'.metrics', package=__package__).__dict__[name.title()]()
            for name in eval_metrics
            }
        self.figures = defaultdict(int) 
        self._results = dict()

        # use optimal threshold (i.e., Youden's J or not)
        if 'youdenj' in self.metric_funcs:
            for func in self.metric_funcs.values():
                if hasattr(func, '_use_youdenj'):
                    setattr(func, '_use_youdenj', True)

    def track(self, loss, pred, true):
        # update running loss
        self.figures['loss'] += loss * len(pred)

        # update running metrics
        for module in self.metric_funcs.values():
            module.collect(pred, true)

    def aggregate(self, total_len, curr_step=None):
        running_figures = {name: module.summarize() for name, module in self.metric_funcs.items()}
        running_figures['loss'] = self.figures['loss'] / total_len
        if curr_step is not None:
            self._results[curr_step] = {
                'loss': running_figures['loss'], 
                'metrics': {name: running_figures[name] for name in self.metric_funcs.keys()}
                }
        else:
            self._results = {
                'loss': running_figures['loss'], 
                'metrics': {name: running_figures[name] for name in self.metric_funcs.keys()}
                }
        self.figures = defaultdict(int)

    @property
    def results(self):
        return self._results
        