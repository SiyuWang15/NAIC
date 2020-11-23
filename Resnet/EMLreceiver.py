import numpy as np
import time 
import multiprocessing as mp
import math
from queue import Queue

def getERH(hf_hat, hf_true, save = False):
    # 输入:
    # hf_hat 信道估计值 频域 N*2*2*256 复数
    # hf_true 信道真实值 频域 N*2*2*256 复数
    # 输出:
    # Edh 估计误差的均值 2*2*256
    # Rdh 估计误差的二阶统计量 16*256
    # 依次为
    # a1a1H a1a2H a1Ha2 a2a2H 
    # a1a3H a1a4H a2a3H a2a4H 
    # a1Ha3 a2Ha3 a1Ha4 a2Ha4 
    # a3a3H a3a4H a3Ha4 a4a4H
    dh = hf_true - hf_hat
    Edh = np.mean(dh, 0)
    Rdh = np.zeros([16, 256], dtype = np.complex128)

    Rdh[0,:] =  np.mean(dh[:,0,0,:] * np.conj(dh[:, 0,0,:]),0) # a1a1H
    Rdh[1,:] =  np.mean(dh[:,0,0,:] * np.conj(dh[:, 0,1,:]),0) # a1a2H      
    Rdh[2,:] =  np.mean(dh[:,0,1,:] * np.conj(dh[:, 0,0,:]),0) # a2a1H
    Rdh[3,:] =  np.mean(dh[:,0,1,:] * np.conj(dh[:, 0,1,:]),0) # a2a2H

    Rdh[4,:] =  np.mean(dh[:,0,0,:] * np.conj(dh[:, 1,0,:]),0) # a1a3H
    Rdh[5,:] =  np.mean(dh[:,0,0,:] * np.conj(dh[:, 1,1,:]),0) # a1a4H 
    Rdh[6,:] =  np.mean(dh[:,0,1,:] * np.conj(dh[:, 1,0,:]),0) # a2a3H  
    Rdh[7,:] =  np.mean(dh[:,0,1,:] * np.conj(dh[:, 1,1,:]),0) # a2a4H  

    Rdh[8,:] =  np.mean(dh[:,1,0,:] * np.conj(dh[:, 0,0,:]),0) # a3a1H
    Rdh[9,:] =  np.mean(dh[:,1,0,:] * np.conj(dh[:, 0,1,:]),0) # a3a2H
    Rdh[10,:] = np.mean(dh[:,1,1,:] * np.conj(dh[:, 0,0,:]),0) # a4a1H 
    Rdh[11,:] = np.mean(dh[:,1,1,:] * np.conj(dh[:, 0,1,:]),0) # a4a2H

    Rdh[12,:] = np.mean(dh[:,1,0,:] * np.conj(dh[:, 1,0,:]),0) #a3a3H
    Rdh[13,:] = np.mean(dh[:,1,0,:] * np.conj(dh[:, 1,1,:]),0) #a3a4H
    Rdh[14,:] = np.mean(dh[:,1,1,:] * np.conj(dh[:, 1,0,:]),0) #a4a3H
    Rdh[15,:] = np.mean(dh[:,1,1,:] * np.conj(dh[:, 1,1,:]),0) #a4a4H

    if save:
        np.save('Rdh.npy', Rdh)
        np.save('Edh.npy', Edh)
    
    return Edh, Rdh


def getx(x):
    x1 = x[0,0]
    x2 = x[0,1]
    x3 = x[1,0]
    x4 = x[1,1]
    return x1,x2,x3,x4

def GetCh(x1,x2,x3,x4,R):
    C = np.zeros([2,2], dtype = np.complex128)
    C[0,0] = x1*R[0] + x2*R[1] + x3*R[2] + x4*R[3]
    C[0,1] = x1*R[4] + x2*R[5] + x3*R[6] + x4*R[7]
    C[1,0] = x1*R[8] + x2*R[9] + x3*R[10] + x4*R[11]
    C[1,1] = x1*R[12] + x2*R[13] + x3*R[14] + x4*R[15]
    return C

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

def EMLReceiver_single_process(q, label, Y, H, Edh, Rdh, SNRdb = 10, Codebook = MakeCodebook(4)):
    # 输入：
    # Y : frequency domain 复数 (batch * 2 * 256) [样本数 * 天线数 * 子载波数]
    # H : frequency domain 估计的信道 复数 (batch * 2 * 2 * 256) [样本数 * 接收天线数 * 发射天线数 * 子载波数]
    # Edh: 估计信道误差的均值 复数 (2*2*256) [接收天线数 * 发射天线数 * 子载波数]
    # Rdh : 估计信道误差的二阶统计量16*256
    # 依次为: a1a1H a1a2H a1Ha2 a2a2H a1a3H a1a4H a2a3H a2a4H a1Ha3 a2Ha3 a1Ha4 a2Ha4 a3a3H a3a4H a3Ha4 a4a4H
    # SNRdb: 信噪比
    # 输出：
    # X_ML : frequency domain 复数 (batch * 2 * 256) [样本数 * 发射天线数 * 子载波数]
    # X_bits    : frequency domain 比特 (batch * 1024) 

    G, P = Codebook.shape
    batch = H.shape[0]

    Y = np.reshape(Y , (batch, 2,  256))
    H = np.reshape(H , (batch, 2,  2, 256))
    X_ML = np.zeros((batch, 2, 256) , dtype = complex)
    X_bits = np.zeros((batch, 2, 2, 256))
    
    sigma2 = 0.35 * 10 ** (-SNRdb / 10)
    Rn = sigma2 * np.eye(2)

    for num in range(batch):
        if num % 100 == 0:
            print('P{0}'.format(label),': Completed batches [%d]/[%d]'%(num ,batch), time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        for idx in range(256):
            y = Y[num, :, idx:idx + 1]
            h = H[num, :, :, idx]
            edh = Edh[:,:,idx]
            error = np.zeros((P, 1)) 
            for item in range(P):

                x = np.reshape(Codebook[:, item], (2,2))
                x = 0.7071 * ( 2 * x[:,0] - 1 ) + 0.7071j * ( 2 * x[:,1] - 1 )
                x = x.reshape([2,1])
                xxH = x.dot(x.T.conj())
                x1,x2,x3,x4 = getx(xxH)
                
                Cdh = GetCh(x1,x2,x3,x4,Rdh[:, idx])
                Chx = Cdh + Rn
                Ehx = np.dot(h, x) + np.dot(edh, x)
                
                error[item] =  np.abs(((y - Ehx).T.conj()).dot(np.linalg.inv(Chx)).dot(y - Ehx))  
            
            ML_idx = np.argmin(error)

            x = np.reshape(Codebook[:, ML_idx], (2,2))
            x_ML = 0.7071 * ( 2 * x[:,0] - 1 ) + 0.7071j * ( 2 * x[:,1] - 1 )
            x = x_ML.reshape([2,1])
            X_ML[num,:,idx] = x_ML

    X_ML = np.reshape(X_ML, [batch, 2 , 256])
    # X_bits = np.reshape(X_bits, [batch, 2, 512])
    # Batch * TX * subcarrier *  IQ
    X_bits = np.reshape(X_ML, [batch, 2 , 256, 1])
    X_bits = np.concatenate([np.real(X_bits)>0,np.imag(X_bits)>0], 3)*1
    X_bits = np.reshape(X_bits, [batch, 1024])

    # q.put([X_ML, X_bits, label])
    # q.put( label )

    q[label] = (X_ML, X_bits)
    print('P{0} completed!'.format(label))



def EMLReceiver(Y, H, Edh, Rdh, SNRdb = 10,  Codebook = MakeCodebook(4), num_workers = 16):
    # 输入：
    # Y : frequency domain 复数 (batch * 2 * 256) [样本数 * 天线数 * 子载波数]
    # H : frequency domain 估计的信道 复数 (batch * 2 * 2 * 256) [样本数 * 接收天线数 * 发射天线数 * 子载波数]
    # Edh: 估计信道误差的均值 复数 (2*2*256) [接收天线数 * 发射天线数 * 子载波数]
    # Rdh : 估计信道误差的二阶统计量16*256
    # 依次为: a1a1H a1a2H a1Ha2 a2a2H a1a3H a1a4H a2a3H a2a4H a1Ha3 a2Ha3 a1Ha4 a2Ha4 a3a3H a3a4H a3Ha4 a4a4H
    # SNRdb: 信噪比
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
            P = mp.Process( target = EMLReceiver_single_process, args = (q, i, Y_single, H_single, Edh, Rdh, SNRdb, Codebook))
            # P.start()
            Processes.append(P)
        else:
            Y_single = Y[  i*batch :, ...]
            H_single = H[  i*batch :, ...]
            P = mp.Process( target = EMLReceiver_single_process, args = (q, i, Y_single, H_single, Edh, Rdh, SNRdb, Codebook))
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