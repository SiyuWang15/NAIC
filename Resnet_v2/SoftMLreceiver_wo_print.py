import numpy as np
import time 
import multiprocessing as mp
import math
from queue import Queue


def MakeCodebook(G = 4):
    codebook = np.zeros((G, 2**G))
    for idx in range(2**G):
        n = idx
        for i in range(G):
            r = n % 2
            codebook[G -1- i, idx] = r
            n = n//2

    return codebook

def SoftMLReceiver_single_process(q, label, Y, H, SNRdb = 10):
    # 输入：
    # Y : frequency domain 复数 (batch * 2 * 256) [样本数 * 天线数 * 子载波数]
    # H : frequency domain 复数 (batch * 2 * 2 * 256) [样本数 * 接收天线数 * 发射天线数 * 子载波数]
    # 输出：
    # X_ML : frequency domain 复数 (batch * 2 * 256) [样本数 * 发射天线数 * 子载波数]
    # X_bits    : frequency domain 比特 (batch * 1024) 


    
    codebook = MakeCodebook()
    G = 4
    

    batch = H.shape[0]


    Y = np.reshape(Y , (batch, 2,  256))
    H = np.reshape(H , (batch, 2,  2, 256))
    X_ML = np.zeros((batch, 2, 256) , dtype = complex)
    X_bits = np.zeros((batch, 2, 2, 256))

    index = [0,1,2,3]

    index_set = [[1,2,3],[0,2,3],[0,1,3],[0,1,2]]

    sigma2 = 0.35 * 10 ** (-SNRdb / 10)
    for num in range(batch):
        for idx in range(256):
            y = Y[num, :, idx:idx+1]
            h = H[num, :, :, idx]
            error = np.zeros((2**G, 1)) 
            x_complex = np.zeros( (2, 1) , dtype = complex)
            x_bit = np.zeros((4,1))
            # 码字遍历
            for idx1 in range(2**G):
                x = np.zeros( (2, 1) , dtype = complex)    
                bits = codebook[:, idx1]   
                x[0] = 0.7071 * ( 2 * bits[0] - 1 ) + 0.7071j * ( 2 * bits[1] - 1 )
                x[1] = 0.7071 * ( 2 * bits[2] - 1 ) + 0.7071j * ( 2 * bits[3] - 1 )
                # bits = np.reshape(codebook[:, idx1], (2,2))
                # x = 0.7071 * ( 2 * bits[:,0] - 1 ) + 0.7071j * ( 2 * bits[:,1] - 1 )
                # x = x.reshape([2,1])
                error[idx1] = np.linalg.norm( y - np.dot(h, x) )**2
            # 软bit计算
            for idx2 in range(G):
                #依靠codebook检索
                code_idx = codebook[idx2, :]
                # error0 = np.sum(error[ code_idx < 0.5 ])
                # error1 = np.sum(error[ code_idx > 0.5 ])
                # if error0 < error1:
                #     x_bit[idx2] = 0.
                # else:
                #     x_bit[idx2] = 1.

                LR0 = np.sum(np.exp( - error[ code_idx < 0.5 ]/sigma2 ))
                LR1 = np.sum(np.exp( - error[ code_idx > 0.5 ]/sigma2 ))
                if LR0 > LR1:
                    x_bit[idx2] = 0.
                else:
                    x_bit[idx2] = 1.

            
            x_complex[0] = 0.7071 * ( 2 * x_bit[0] - 1 ) + 0.7071j * ( 2 * x_bit[1] - 1 )
            x_complex[1] = 0.7071 * ( 2 * x_bit[2] - 1 ) + 0.7071j * ( 2 * x_bit[3] - 1 )
            # ML_idx = np.argmin(error)
            # bits = np.reshape(codebook[:, ML_idx], (2,2))
            # x_complex = 0.7071 * ( 2 * bits[:,0] - 1 ) + 0.7071j * ( 2 * bits[:,1] - 1 )
            # x_complex = x_complex.reshape([2,1])
            X_ML[num,:,idx:idx+1] = x_complex 

    X_bits = np.reshape(X_ML, [batch, 2 , 256, 1])
    X_bits = np.concatenate([np.real(X_bits)>0,np.imag(X_bits)>0], 3)*1
    X_bits = np.reshape(X_bits, [batch, 1024])

    q[label] = (X_ML, X_bits)

def SoftMLReceiver(Y, H, Codebook = MakeCodebook(4), SNRdb = 10, num_workers = 16):
    # 输入：
    # Y : frequency domain 复数 (batch * 2 * 256) [样本数 * 天线数 * 子载波数]
    # H : frequency domain 复数 (batch * 2 * 2 * 256) [样本数 * 接收天线数 * 发射天线数 * 子载波数]
    # 输出：
    # X_ML : frequency domain 复数 (batch * 2 * 256) [样本数 * 发射天线数 * 子载波数]
    # X_bits    : frequency domain 比特 (batch * 1024) 

    G, P = Codebook.shape
    assert P == 2**G


    N_s = H.shape[0]


    # 创建多进程
    q = mp.Manager().dict()
    Processes = []
    batch = math.floor(N_s * 1. / num_workers)

    for i in range(num_workers):
        if i < num_workers - 1:
            Y_single = Y[  i*batch : (i+1)*batch, ...]
            H_single = H[  i*batch : (i+1)*batch, ...]
            P = mp.Process( target = SoftMLReceiver_single_process, args = (q, i, Y_single, H_single,SNRdb ))
            # P.start()
            Processes.append(P)
        else:
            Y_single = Y[  i*batch :, ...]
            H_single = H[  i*batch :, ...]
            P = mp.Process( target = SoftMLReceiver_single_process, args = (q, i, Y_single, H_single,SNRdb))
            # P.start()
            Processes.append(P)


    for i in range(num_workers):
        Processes[i].start()
    
    for i in range(num_workers):
        Processes[i].join()


    X_ML = np.zeros((N_s, 2, 256) , dtype = complex)
    X_bits = np.zeros((N_s, 1024))
    
    for label,v in q.items():
        ml, bits = v
        if label < num_workers - 1:
            X_ML[  label*batch : (label+1)*batch, :, :] = ml
            X_bits[  label*batch : (label+1)*batch, :] = bits
        else:
            X_ML[  label*batch: , :, :] = ml
            X_bits[  label*batch: , :] = bits

    return X_ML, X_bits

if __name__ == '__main__':

    codebook = MakeCodebook()

    print('Love Live!')
