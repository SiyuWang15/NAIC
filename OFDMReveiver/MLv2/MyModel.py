# import tensorflow as tf
# from tensorflow import keras
from utils import *
# import  torch as t
import struct
import numpy as np
import torch
from MLreceiver import *

mode=2
SNRdb=10
Pilotnum=32
'''
model = keras.MyModel() #定义自己的模型
model.summary()
model.compile(
    optimizer=keras.optimizers.Adam(),  # Optimizer
    # Loss function to minimize
    loss='mean_squared_error',
    # List of metrics to monitor
    metrics=['binary_accuracy'],
)
'''
####################使用链路和信道数据产生训练数据##########
def generator(batch,H):
    while True:
        input_labels = []
        input_samples = []
        for row in range(0, batch):
            bits0 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
            bits1 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
            X=[bits0, bits1]
            temp = np.random.randint(0, len(H))
            HH = H[temp]
            YY = MIMO(X, HH, SNRdb, mode,Pilotnum)/20 ###
            XX = np.concatenate((bits0, bits1), 0)
            input_labels.append(XX)
            input_samples.append(YY)
        batch_y = np.asarray(input_samples)
        batch_x = np.asarray(input_labels)
        yield (batch_y, batch_x)
#####训练#########
# model.fit_generator(generator(1000,H),steps_per_epoch=50,epochs=2000)

########产生测评数据，仅供参考格式##########
def generatorXY(batch, H):
    input_labels = []
    input_samples = []
    input_channels = []

    for row in range(0, batch):
        bits0 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
        bits1 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
        X = [bits0, bits1]
        temp = np.random.randint(0, len(H))
        HH = H[temp]
        YY = MIMO(X, HH, SNRdb, mode, Pilotnum) / 20  ###
        XX = np.concatenate((bits0, bits1), 0)
        input_labels.append(XX)
        input_samples.append(YY)
        input_channels.append(HH)

    batch_y = np.asarray(input_samples)
    batch_x = np.asarray(input_labels)
    batch_h = np.asarray(input_channels)
    return batch_y, batch_x, batch_h

if __name__ == '__main__':
    ###########################以下仅为信道数据载入和链路使用范例############

    data1=open('/data/CuiMingyao/AI_competition/OFDMReceiver/H.bin','rb')
    H1=struct.unpack('f'*2*2*2*32*320000, data1.read(4*2*2*2*32*320000))
    H1=np.reshape(H1,[320000,2,4,32])
    H=H1[:,1,:,:]+1j*H1[:,0,:,:]

    Htest=H[300000:,:,:]
    Htrain=H[:300000,:,:]

    batch = 20
    Y, X, H = generatorXY(batch, Htrain)
    
    # np.savetxt('./Y_1.csv', Y, delimiter=',')
    # X_1 = np.array(np.floor(X + 0.5), dtype=np.bool)
    # X_1.tofile('./X_1.bin')


    #####################数据维度说明####################

    ### 接受信号维度说明
    ### 这里一定要加order！！！！######
    # 样本数 * 虚实部 * 导频and数据 * 天线数 * 子载波数
    output = np.reshape(Y, (-1, 2,2,2, 256), order = 'F') 

    ###例：提取复数数据
    # 输出维度为 样本数 * 天线数 * 子载波数
    output_complex = output[:,0,:,:,:] + 1j*output[:,1,:,:,:]
    output_data = output_complex[:,1,:,:]


    ### 信道维度说明，这里一定要加order！！！！！
    # H:时域信道，样本数 * 天线数(4，依次为第一根Tx至第一根Rx“00”、“01”、“10”、“11” ) * 时域维度(32)
    Hf = np.fft.fft(H, 256)/20
    # Hf：频域信道，样本数 * Rx * Tx * 频域维度
    # Rx * Tx 为 2*2 的矩阵 从上至下，从左至右依次为第一根Tx至第一根Rx、第一根Tx至第二根Rx、第二根Tx至第一根Rx，第二根Tx至第二根Rx
    # 因为MIMO接受数据有除以20，这里就除以了20
    Hf = np.reshape(Hf, (-1, 2,2,256), order='F')



    #### 输入数据维度说明，这里一定【不要】加order！！！！！！
    # X：样本数 * 发射天线数 * （子载波数 \times 虚实部）
    X_temp = np.reshape(X, (-1, 2,512))

    # 可用的复数input：样本数 * 发射天线数 * 子载波数
    input = np.zeros((batch, 2,256), dtype=complex)
    for num in range(batch):
        for idx in range(2):
            bits = X_temp[num, idx, :]
            # 与 utils文件里的modulation一模一样的操作
            bits = np.reshape(bits, (256,2))
            input[num, idx,:] = 0.7071 * (2 * bits[:, 0] - 1) + 0.7071j * (2 * bits[:, 1] - 1)

    # 测试是否满足 Y - HX = 0
    # 各数据维度总结【均为复数】
    # input： batch * 2 * 256    【 样本数 * 发射天线数 * 子载波数 】  
    # Hf：    batch * 2 * 2 *256 【 样本数 * 收端天线数 * 发端天线数 *子载波数 】  
    # output_data 或 Y_hat：batch * 2 * 256 【 样本数 * 收端天线数 * 子载波数 】  
    Y_hat = np.zeros(output_data.shape, dtype=complex)
    for num in range(batch):
        for idx in range(256):
            Y_hat[num, :,idx] = Hf[num, :,:,idx].dot(input[num, :,idx])
    error_Y = output_data - Y_hat

    # print('Hello world')
    codebook = MakeCodebook(4)
    X_ML, X_bits = MLReceiver(output_data, Hf, codebook)
    error_X = input - X_ML
    error_X_bit = X - X_bits

    print('Complex Accuracy: ', np.sum(np.abs(error_X)<0.1)/np.size(error_X)*100,'%')
    print('Bits Accuracy: ', np.sum(np.abs(error_X_bit)<0.1)/np.size(error_X_bit)*100,'%')
    # print('Love live')


