import numpy as np
import time 
import multiprocessing as mp
import math
from queue import Queue

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

if __name__ == '__main__':

    x = np.random.randn(3,2,1,256) + 1j*np.random.randn(3,2,1,256)
    H = np.random.randn(3,2,2,256) + 1j*np.random.randn(3,2,2,256)
    y = np.zeros([3,2,1,256], dtype = np.complex64)
    for idx1 in range(3):
        for idx2 in range(256):   
            y[idx1, :,:, idx2] = H[idx1,:,:,idx2].dot(x[idx1,:,:,idx2])
    y = y.reshape(3,2,256)
    x_hat = LSequalization(H, y)

    print('lovelive')