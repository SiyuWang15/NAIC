import numpy as np 
import random
import math 
import torch
import datetime
from torch.utils.data import Dataset
from .Communication import MIMO
import time
from utils import *

class RandomDataset():
    def __init__(self, H, Pilot_num, SNRdb=-1, mode=0):
        super().__init__()
        self.H = H
        self.Pilot_num = Pilot_num
        self.SNRdb = SNRdb
        self.mode = mode

    def __getitem__(self, index):
        HH = self.H[index]
        seed = math.floor(math.modf(time.time())[0]*500*320000)**2 % (2**32 - 2)
        np.random.seed(seed)
        # print(seed)
        # print(np.random.randn())
        bits0 = np.random.binomial(1, 0.5, size=(128 * 4,))
        bits1 = np.random.binomial(1, 0.5, size=(128 * 4,))
        SS = self.SNRdb
        mm = self.mode
        if self.SNRdb == -1:
            SS =  np.random.uniform(8, 12)
        if self.mode == -1:
            mm = np.random.randint(0, 3)
        YY = MIMO([bits0, bits1], HH, SS, mm, self.Pilot_num)/20
        XX = np.concatenate([bits0, bits1], 0)
        newHH = np.stack([HH.real, HH.imag], axis = 0)
        return YY, XX, newHH

    def __len__(self):
        return len(self.H)
        # return 2000