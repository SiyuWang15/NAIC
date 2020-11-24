import numpy as np 
import torch
import math
import multiprocessing as mp
import os
import time


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
        # if num % 100 == 0:
        #     print('Completed batches [%d]/[%d]'%(num ,batch), time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
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
    # print('P{0} completed!'.format(label))

def SoftMLReceiver(Y, H, Codebook = MakeCodebook(4), SNRdb = 7., num_workers = 16):
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

def MLReceiver(Y, H, num_workers = 16):
    Codebook = MakeCodebook(4)
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
    
def MLReceiver_single_process(q, label, Y, H, Codebook):
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
        # if num % 100 == 0:
        #     print('P{0}'.format(label),': Completed batches [%d]/[%d]'%(num ,batch), time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

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
    X_bits = np.reshape(X_ML, [batch, 2 , 256, 1])
    X_bits = np.concatenate([np.real(X_bits)>0,np.imag(X_bits)>0], 3)*1
    X_bits = np.reshape(X_bits, [batch, 1024])
    q[label] = (X_ML, X_bits)
        # print('P{0} completed!'.format(label))



def MakeCodebookML(G = 4):

    assert type(G) == int

    codebook = np.zeros((G, 2**G))
    for idx in range(2**G):
        n = idx
        for i in range(G):
            r = n % 2
            codebook[G -1- i, idx] = r
            n = n//2

    return codebook

def MLPilot_single_process(q, label, Y, H, Pilot_num, Codebook = MakeCodebook(2)):
    # 输入：
    # Y : frequency domain 复数 (batch * 2 * 256) [样本数 * 天线数 * 子载波数]
    # H : frequency domain 复数 (batch * 2 * 2 * 256) [样本数 * 接收天线数 * 发射天线数 * 子载波数]
    # 输出：
    # X_ML : frequency domain 复数 (batch * 2 * 256) [样本数 * 发射天线数 * 子载波数]
    # X_bits    : frequency domain 比特 (batch * 1024) 

    batch = H.shape[0]
    K = 256
    P = Pilot_num * 2
    allCarriers = np.arange(K)
    pilotCarriers = np.arange(0, K, K // P)
    dataCarriers = [val for val in allCarriers if not (val in pilotCarriers)]
    pilotCarriers1 = pilotCarriers[0:P:2]
    pilotCarriers2 = pilotCarriers[1:P:2]

    Yp1 = Y[:, : , pilotCarriers1]
    Yp2 = Y[:, : , pilotCarriers2]
    Hp1 = H[:, : , :, pilotCarriers1]
    Hp2 = H[:, : , :, pilotCarriers2]


    X_ML1 = np.zeros((batch,  Pilot_num) , dtype = complex)
    X_ML2 = np.zeros((batch,  Pilot_num) , dtype = complex)
#     print(label,'X_ML1:', X_ML1.shape)

    for num in range(batch):
#         if num % 100 == 0:
#             print('P{0} Antenna0'.format(label),': Completed batches [%d]/[%d]'%(num ,batch), time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        for idx in range(Pilot_num):
            y = Yp1[num, :,  idx].reshape([ 2, 1])
            h = Hp1[num, :,  :, idx]
            error = np.zeros((4, 1)) 
            for item in range(4):

                x = Codebook[:, item]
                x = 0.7071 * ( 2 * x[0] - 1 ) + 0.7071j * ( 2 * x[1] - 1 )
                input = np.array([x, 0]).reshape([2,1])
                error[item] = error[item] + np.linalg.norm( y - np.dot(h, input) )**2
            
            ML_idx = np.argmin(error)

            x = Codebook[:, ML_idx]
            x_ML = 0.7071 * ( 2 * x[0] - 1 ) + 0.7071j * ( 2 * x[1] - 1 )
            X_ML1[num, idx] = x_ML

    for num in range(batch):
        if num % 100 == 0:
            print('P{0} Antenna1'.format(label),': Completed batches [%d]/[%d]'%(num ,batch), time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        for idx in range(Pilot_num):
            y = Yp2[num, :,  idx].reshape([ 2, 1])
            h = Hp2[num, :,  :, idx]
            error = np.zeros((4, 1)) 
            for item in range(4):

                x = Codebook[:, item]
                x = 0.7071 * ( 2 * x[0] - 1 ) + 0.7071j * ( 2 * x[1] - 1 )
                input = np.array([0, x]).reshape([2,1])
                error[item] = error[item] + np.linalg.norm( y - np.dot(h, input) )**2
            
            ML_idx = np.argmin(error)

            x = Codebook[:, ML_idx]
            x_ML = 0.7071 * ( 2 * x[0] - 1 ) + 0.7071j * ( 2 * x[1] - 1 )
            X_ML2[num, idx] = x_ML

    Pilot_value = np.zeros((batch, 2*Pilot_num) , dtype = complex)
#     print(label, 'Pilot_ value:', Pilot_value.shape)
    Pilot_value[:,  0:P:2] = X_ML1
    Pilot_value[:,  1:P:2] = X_ML2

    Pilot_bits = np.reshape(Pilot_value, [batch, 2*Pilot_num, 1])
    Pilot_bits = np.concatenate([np.real(Pilot_bits)>0,np.imag(Pilot_bits)>0], 2)*1
    Pilot_bits = np.reshape(Pilot_bits, [batch, 2*2*Pilot_num])


    q[label] = (Pilot_value, Pilot_bits)
#     print('P{0} completed!'.format(label))



def MLPilot(Y, H, Pilot_num = 8, Codebook = MakeCodebookML(2), num_workers = 8):
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
            P = mp.Process( target = MLPilot_single_process, args = (q, i, Y_single, H_single,Pilot_num,Codebook))
            # P.start()
            Processes.append(P)
        else:
            Y_single = Y[  i*batch :, ...]
            H_single = H[  i*batch :, ...]
            P = mp.Process( target = MLPilot_single_process, args = (q, i, Y_single, H_single,Pilot_num, Codebook))
            # P.start()
            Processes.append(P)


    for i in range(num_workers):
        Processes[i].start()
    
    for i in range(num_workers):
        Processes[i].join()


    X_ML = np.zeros((N_s, 2*Pilot_num) , dtype = complex)
    X_bits = np.zeros((N_s, 2*2*Pilot_num))
    
    for label,v in q.items():
        ml, bits = v
        if label < num_workers - 1:
            X_ML[  label*batch : (label+1)*batch, :] = ml
            X_bits[  label*batch : (label+1)*batch , :] = bits
        else:
            X_ML[  label*batch: , :] = ml
            X_bits[  label*batch: , :] = bits

    return X_ML, X_bits