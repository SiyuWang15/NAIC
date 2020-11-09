r'''to be implemented'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np 
import logging
import sys
sys.path.append('..')
from Estimators import Y2XEstimator
from get_data import get_YX_data

class Y2XRunner():
    def __init__(self, config):
        self.config = config
    
    def get_optimizer(self, parameters):
        if self.config.train.optimizer == 'adam':
            return torch.optim.Adam(parameters, lr = self.config.train.lr)
        elif self.config.train.optimizer == 'sgd':
            return torch.optim.SGD(parameters, lr = self.config.train.lr, momentum=0.9)
        else:
            raise NotImplementedError('Optimizer {} not understood.'.format(self.config.train.optimizer))
    
    def run(self):
        pass