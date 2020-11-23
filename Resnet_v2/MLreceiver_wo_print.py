import numpy as np
import time 
import multiprocessing as mp
import math
from queue import Queue

def MakeCodebook(G = 4):

    assert type(G) == int

    codebook = np.zeros((G, 2**G))
    for idx in range(2**G):
        n = idx
        for i in range(G):
            r = n % 2
            codebook[G -1- i, idx] = r
            n = n//2

    return codebook

def MLReceiver_single_process(q, label, Y, H, Codebook = MakeCodebook(4)):
    # 输入：
    # Y : frequency domain 复数 (batch * 2 * 256) [样本数 * 天线数 * 子载波数]
    # H : frequency domain 复数 (batch * 2 * 2 * 256) [样本数 * 接收天线数 * 发射天线数 * 子载波数]
    # 输出：
    # X_ML : frequency domain 复数 (batch * 2 * 256) [样本数 * 发射天线数 * 子载波数]
    # X_bits    : frequency domain 比特 (batch * 1024) 

    G, P = Codebook.shape
    B = G//4
    T = 256//B
    batch = H.shape[0]

    # B为分组的个数，只能为2的整倍数
    Y = np.reshape(Y , (batch, 2,  B, T))
    H = np.reshape(H , (batch, 2,  2, B, T))
    X_ML = np.zeros((batch, 2, B, T) , dtype = complex)
    X_bits = np.zeros((batch, 2, 2, B, T))

    for num in range(batch):
        for idx in range(T):
            y = Y[num, :, :, idx]
            h = H[num, :, :, :, idx]
            error = np.zeros((P, 1)) 
            for item in range(P):

                x = np.reshape(Codebook[:, item], (2,2,B))
                x = 0.7071 * ( 2 * x[:,0,:] - 1 ) + 0.7071j * ( 2 * x[:,1,:] - 1 )
                for b in range(B):
                    error[item] = error[item] + np.linalg.norm( y[:,b:b+1] - np.dot(h[:,:,b], x[:,b:b+1]) )**2
            
            ML_idx = np.argmin(error)

            x = np.reshape(Codebook[:, ML_idx], (2,2,B))
            x_ML = 0.7071 * ( 2 * x[:,0,:] - 1 ) + 0.7071j * ( 2 * x[:,1,:] - 1 )
            X_ML[num,:,:,idx] = x_ML

    X_ML = np.reshape(X_ML, [batch, 2 , 256])
    # X_bits = np.reshape(X_bits, [batch, 2, 512])
    # Batch * TX * subcarrier *  IQ
    X_bits = np.reshape(X_ML, [batch, 2 , 256, 1])
    X_bits = np.concatenate([np.real(X_bits)>0,np.imag(X_bits)>0], 3)*1
    X_bits = np.reshape(X_bits, [batch, 1024])

    # q.put([X_ML, X_bits, label])
    # q.put( label )

    q[label] = (X_ML, X_bits)



def MLReceiver(Y, H, Codebook = MakeCodebook(4), num_workers = 16):
    # 输入：
    # Y : frequency domain 复数 (batch * 2 * 256) [样本数 * 天线数 * 子载波数]
    # H : frequency domain 复数 (batch * 2 * 2 * 256) [样本数 * 接收天线数 * 发射天线数 * 子载波数]
    # 输出：
    # X_ML : frequency domain 复数 (batch * 2 * 256) [样本数 * 发射天线数 * 子载波数]
    # X_bits    : frequency domain 比特 (batch * 1024) 

    G, P = Codebook.shape
    assert P == 2**G

    B = G//4
    assert B*4 == G

    T = 256//B
    assert T*B == 256

    N_s = H.shape[0]


    # 创建多进程
    q = mp.Manager().dict()
    Processes = []
    batch = math.floor(N_s * 1. / num_workers)

    for i in range(num_workers):
        if i < num_workers - 1:
            Y_single = Y[  i*batch : (i+1)*batch, ...]
            H_single = H[  i*batch : (i+1)*batch, ...]
            P = mp.Process( target = MLReceiver_single_process, args = (q, i, Y_single, H_single,Codebook))
            # P.start()
            Processes.append(P)
        else:
            Y_single = Y[  i*batch :, ...]
            H_single = H[  i*batch :, ...]
            P = mp.Process( target = MLReceiver_single_process, args = (q, i, Y_single, H_single,Codebook))
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

def test(a):
    a = np.reshape(a, [-1,2])
    print(a)

if __name__ == '__main__':
    
    a = np.arange(24)
    print(a)
    test(a)
    print('Love Live!')
    print(a)