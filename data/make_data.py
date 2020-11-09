import numpy as np 
import random
from H_utils import *

data_prefix = '/data/siyu/NAIC'
H_path = os.path.join(data_prefix, 'dataset/H_data.npy')

def make_Ypmode_data():
    H = np.load(H_path) # Nsx4x32 complex numbers 
    Yp2mode = []
    for i in range(len(H)):
        HH = H[i, :, :]
        mode = random.randint(0, 2)
        SNRdb = random.randint(8, 12)
        bits0 = np.random.binomial(1, 0.5, size=(128*4, ))
        bits1 = np.random.binomial(1, 0.5, size=(128*4, ))
        YY = MIMO([bits0, bits1], HH, SNRdb, mode, 32) / 20
        YY = np.reshape(YY, [2, 2, 2, 256], order='F')
        Yp = YY[:, 0, :, :].reshape(1024, order = 'F')
        Yp2mode.append((Yp, mode))
        if i % 10000 == 0:
            print('%d complete.' % i)
    
    np.save(os.path.join(data_prefix, 'dataset/random_mode/Yp2mode_Pilot32.npy', Yp2mode, allow_pickle=True)

def make_YH_data():
    H = np.load(H_path)
    Pilotnum = 32
    Yp_modes = [[], [], []]
    for i in range(len(H)):
        HH = H[i, :, :]
        SNRdb = random.randint(8, 12)
        bits0 = np.random.binomial(1, 0.5, size=(128*4, ))
        bits1 = np.random.binomial(1, 0.5, size=(128*4, ))
        
        YY = MIMO([bits0, bits1], HH, SNRdb, 0, Pilotnum) / 20
        YY = np.reshape(YY, [2, 2, 2, 256], order = 'F')
        Yp = YY[:, 0, :, :].reshape(1024, order = 'F')
        Yp_modes[0].append(Yp)

        YY = MIMO([bits0, bits1], HH, SNRdb, 1, Pilotnum) / 20
        YY = np.reshape(YY, [2, 2, 2, 256], order = 'F')
        Yp = YY[:, 0, :, :].reshape(1024, order = 'F')
        Yp_modes[1].append(Yp)

        YY = MIMO([bits0, bits1], HH, SNRdb, 2, Pilotnum) / 20
        YY = np.reshape(YY, [2, 2, 2, 256], order = 'F') 
        Yp = YY[:, 0, :, :].reshape(1024, order = 'F')
        Yp_modes[2].append(Yp)
        
        if i % 10000 == 0:
            print('%d complete.' % i)
    for mode in [0, 1, 2]:
        np.save(os.path.join(data_prefix, 'dataset/YHdata/Yp_mode_{}_P_{}.npy'.format(mode, Pilotnum), Yp_modes[mode]))

if __name__ == "__main__":
    make_YH_data()