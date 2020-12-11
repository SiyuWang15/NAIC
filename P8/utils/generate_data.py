import sys 
sys.path.append('../')
from utils.utils import *
import struct
import numpy as np
import time
####################使用链路和信道数据产生训练数据##########
def generator(batch,H,Pilot_num, SNR,m):
    while True:
        input_labels = []
        input_samples = []
        input_channels = []
        for _ in range(0, batch):
            bits0 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
            bits1 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
            X=[bits0, bits1]
            temp = np.random.randint(0, len(H))
            HH = H[temp]

            if SNR == -1:
                SNRdb = np.random.uniform(8, 12)
            else:
                SNRdb = SNR

            if m == -1:
                mode = np.random.randint(0, 3)
            else:
                mode = m
            YY = MIMO(X, HH, SNRdb, mode,Pilot_num)/20 ###
            XX = np.concatenate((bits0, bits1), 0)
            input_labels.append(XX)
            input_samples.append(YY)
            input_channels.append(HH)
        batch_y = np.asarray(input_samples)
        batch_x = np.asarray(input_labels)
        batch_h = np.asarray(input_channels)
        yield (batch_y, batch_x, batch_h)

########产生测评数据，仅供参考格式##########
def generatorXY(batch, H, Pilot_num,SNR,m):
    input_labels = []
    input_samples = []
    input_channels = []
    for row in range(0, batch):
        bits0 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
        bits1 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
        X = [bits0, bits1]
        # temp = np.random.randint(0, len(H))
        temp = row
        HH = H[temp]
        if SNR == -1:
            SNRdb = np.random.uniform(8, 12)
        else:
            SNRdb = SNR
            
        if m == -1:
            mode = np.random.randint(0, 3)
        else:
            mode = m
        # print(SNRdb, mode)
        YY = MIMO(X, HH, SNRdb, mode, Pilot_num) / 20  ###
        XX = np.concatenate((bits0, bits1), 0)
        input_labels.append(XX)
        input_samples.append(YY)
        input_channels.append(HH)

    batch_y = np.asarray(input_samples)
    batch_x = np.asarray(input_labels)
    batch_h = np.asarray(input_channels)
    return batch_y, batch_x, batch_h





class RandomDataset():
    def __init__(self, H, Pilot_num, SNRdb=-1, mode=-1):
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
        return YY, XX, HH

    def __len__(self):
        return len(self.H)
        # return 2000


