import numpy as np
import time 
import multiprocessing as mp
import math
from queue import Queue
import torch
import sys 
sys.path.append('../')
def LSequalization(h, y):
    # y 复数： batch*2*256
    # h 复数： batch*2*2*256
    batch = h.shape[0]

    norm = h[:,0,0,:] * h[:,1,1,:] - h[:,0,1,:] * h[:,1,0,:]
    norm = norm.reshape([batch,1,1,256])
    invh = np.zeros([batch,2,2,256], dtype = np.complex64)
    invh[:,0,0,:] = h[:,1,1,:]
    invh[:,1,1,:] = h[:,0,0,:]
    invh[:,0,1,:] = -h[:,0,1,:]
    invh[:,1,0,:] = -h[:,1,0,:]
    invh = invh/norm


    y = y.reshape(batch, 1, 2, 256)
    
    x_LS = invh*y
    x_LS = x_LS[:,:,0,:] + x_LS[:,:,1,:]
    return x_LS

def get_LS(Yd_input, Hf):
    Yd = np.array(Yd_input[:,0,:,:] + 1j*Yd_input[:,1,:,:])
    Hf = np.array(Hf[:,0,:,:] + 1j*Hf[:,1,:,:]) 
    Hf = np.reshape(Hf, [-1,2,2,256], order = 'F')
    X_LS = LSequalization(Hf, Yd)
    X_LS.real = (X_LS.real > 0)*2 - 1
    X_LS.imag = (X_LS.imag > 0)*2 - 1
    return X_LS

if __name__ == '__main__':

    # y = torch.randn(200,2,2,256)
    # H = torch.randn(200,2,4,256)
    # y = np.random.randn(200,2,2,256)
    # H = np.random.randn(200,2,4,256)
    # print(H[0,0,0,0:20])
    # X = get_LS(y,H)
    # print(H[0,0,0,0:20])
    x = np.random.randn(3,2,1,256) + 1j*np.random.randn(3,2,1,256)
    H = np.random.randn(3,2,2,256) + 1j*np.random.randn(3,2,2,256)
    y = np.zeros([3,2,1,256], dtype = np.complex64)
    for idx1 in range(3):
        for idx2 in range(256):   
            y[idx1, :,:, idx2] = H[idx1,:,:,idx2].dot(x[idx1,:,:,idx2])
    y = y.reshape(3,2,256)
    x_hat = LSequalization(H, y)

    print('lovelive')