import numpy as np 
import random
import struct

from Communication import *

dataset_prefix = '/data/siyu/NAIC'
H_path = os.path.join(dataset_prefix, 'dataset/H_data.npy')

def make_Ypmode_data():
    H = np.load(H_path) # Nsx4x32 complex numbers 
    Pn = 8
    Yp2mode = []
    for i in range(len(H)):
        HH = H[i, :, :]
        # mode = 0 if np.random.rand()<0.5 else 2
        mode = random.randint(0, 2)
        SNRdb = np.random.uniform(8, 12)
        bits0 = np.random.binomial(1, 0.5, size=(128*4, ))
        bits1 = np.random.binomial(1, 0.5, size=(128*4, ))
        YY = MIMO([bits0, bits1], HH, SNRdb, mode, Pn) / 20
        YY = np.reshape(YY, [2, 2, 2, 256], order='F')
        Yp = YY[:, 0, :, :].reshape(1024, order = 'F')
        Yp2mode.append((Yp, mode))
        if i % 10000 == 0:
            print('%d complete.' % i)
    
    np.save(os.path.join(dataset_prefix, f'dataset/YModeData/Yp2mode_Pilot{Pn}.npy'), Yp2mode, allow_pickle=True)

def make_YH_data():
    H = np.load(H_path)
    Pilotnum = 32
    Yp_modes = [[], [], []]
    for i in range(len(H)):
        HH = H[i, :, :]
        bits0 = np.random.binomial(1, 0.5, size=(128*4, ))
        bits1 = np.random.binomial(1, 0.5, size=(128*4, ))
        
        for mode in [0,1,2]:
            SNRdb = np.random.uniform(8, 12)
            YY = MIMO([bits0, bits1], HH, SNRdb, mode, Pilotnum) / 20.
            YY = np.reshape(YY, [2, 2, 2, 256], order = 'F')
            Yp = YY[:, 0, :, :].reshape(1024, order = 'F')
            Yp_modes[mode].append(Yp)
        
        # YY = MIMO([bits0, bits1], HH, SNRdb, 0, Pilotnum) / 20
        # YY = np.reshape(YY, [2, 2, 2, 256], order = 'F')
        # Yp = YY[:, 0, :, :].reshape(1024, order = 'F')
        # Yp_modes[0].append(Yp)

        # YY = MIMO([bits0, bits1], HH, SNRdb, 1, Pilotnum) / 20
        # YY = np.reshape(YY, [2, 2, 2, 256], order = 'F')
        # Yp = YY[:, 0, :, :].reshape(1024, order = 'F')
        # Yp_modes[1].append(Yp)

        # YY = MIMO([bits0, bits1], HH, SNRdb, 2, Pilotnum) / 20
        # YY = np.reshape(YY, [2, 2, 2, 256], order = 'F') 
        # Yp = YY[:, 0, :, :].reshape(1024, order = 'F')
        # Yp_modes[2].append(Yp)
        
        if i % 10000 == 0:
            print('%d complete.' % i)
    for mode in [0, 1, 2]:
        np.save(os.path.join(dataset_prefix, 'dataset/YHData/Yp_mode_{}_P_{}.npy'.format(mode, Pilotnum), Yp_modes[mode]))
    
def make_test_data():
    H_path = os.path.join(dataset_prefix, 'dataset/H_val.bin')
    H_data = open(H_path, 'rb')
    H = struct.unpack('f'*2*2*2*32*2000, H_data.read(4*2*2*2*32*2000))
    H = np.reshape(H, [2000, 2, 4, 32]).astype('float32')
    H = H[:, 1, :, :] + 1j*H[:, 0, :, :]
    Y = []
    X = []
    Pn = 32
    for i in range(len(H)):
        SNRdb = random.randint(8, 12)
        mode = 0 if np.random.rand()<0.5 else 2
        bits0 = np.random.binomial(1, 0.5, size = (128*4, ))
        bits1 = np.random.binomial(1, 0.5, size = (128*4, ))
        HH = H[i, :, :]
        YY = MIMO([bits0, bits1], HH, SNRdb, mode, Pn) / 20.
        Y.append(YY)
        X.append(np.concatenate([bits0, bits1], 0))
    Y = np.stack(Y, axis = 0).astype('float32')
    X = np.stack(X, axis = 0).astype('float32')
    X = np.array(np.floor(X+0.5), dtype=np.bool)
    print(X.shape, Y.shape)
    tag = 1 if Pn == 32 else 2
    np.savetxt(os.path.join(dataset_prefix, f'dataset/test/Y_{tag}.csv'), Y, delimiter=',')
    X.tofile(os.path.join(dataset_prefix, f'dataset/test/X_label_{tag}.bin'))



if __name__ == "__main__":
    make_Ypmode_data()