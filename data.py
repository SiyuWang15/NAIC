import numpy as np 
import torch
from H_utils import *
import struct

Pilotnums = [32, 8]
modes = [0,1,2]
SNRdbs = [8, 9, 10, 11, 12]
H_ind = 0

def get_H(H_path = 'dataset/H.bin', ratio = 0.9):
    data = open(H_path, 'rb')
    H = struct.unpack('f'*2*2*2*32*320000,data.read(4*2*2*2*32*320000))
    H = np.reshape(H, [320000, 2, 4, 32])
    H = H[:, 1, :, :] + 1j*H[:, 0, :, :]
    split = int(ratio * 320000)
    H_train = H[:split, :, :]
    H_val = H[split:, :, :]
    return H_train, H_val


def get_data(batch_size, H_stack, device):
    # input_labels = []
    # input_samples = []
    # for row in range(batch_size):
    #     bits0 = np.random.binomial(n=1, p=0.5, size = (128*4, ))
    #     bits1 = np.random.binomial(n=1, p=0.5, size = (128*4, ))
    #     X = [bits0, bits1]
    #     # H_ind = np.random.randint(0, len(H_stack))
    #     HH = H_stack[H_ind]
    #     mode = 0
    #     SNRdb = 12
    #     Pilotnum = 32
    #     # mode = np.random.choice(modes, 1)[0]
    #     # SNRdb = np.random.choice(SNRdbs, 1)[0]
    #     # Pilotnum = np.random.choice(Pilotnums, 1)[0]
    #     YY = MIMO(X, HH, SNRdb, mode, Pilotnum) / 20
    #     YY = np.reshape(YY, [2, 2, 2, 256], order = 'F')
    #     XX = np.concatenate((bits0, bits1), 0)
    #     input_labels.append(XX)
    #     input_samples.append(YY)
    # batch_y = torch.Tensor(np.asarray(input_samples)).to(device)
    # batch_x = torch.Tensor(np.asarray(input_labels)).to(device)
    while True:
        input_labels = []
        input_samples = []
        for row in range(batch_size):
            bits0 = np.random.binomial(n=1, p=0.5, size = (128*4, ))
            bits1 = np.random.binomial(n=1, p=0.5, size = (128*4, ))
            X = [bits0, bits1]
            # H_ind = np.random.randint(0, len(H_stack))
            HH = H_stack[H_ind]
            mode = 0
            SNRdb = 25
            Pilotnum = 32
            # mode = np.random.choice(modes, 1)[0]
            # SNRdb = np.random.choice(SNRdbs, 1)[0]
            # Pilotnum = np.random.choice(Pilotnums, 1)[0]
            YY = MIMO(X, HH, SNRdb, mode, Pilotnum) / 20
            YY = np.reshape(YY, [2, 2, 2, 256], order = 'F')
            XX = np.concatenate((bits0, bits1), 0)
            input_labels.append(XX)
            input_samples.append(YY)
        batch_y = torch.Tensor(np.asarray(input_samples)).to(device)
        batch_x = torch.Tensor(np.asarray(input_labels)).to(device)
        yield (batch_y, batch_x)

def get_val_data(batch_size, H_stack, device):
    input_labels = []
    input_samples = []
    for row in range(batch_size):
        bits0 = np.random.binomial(n=1, p=0.5, size = (128*4, ))
        bits1 = np.random.binomial(n=1, p=0.5, size = (128*4, ))
        X = [bits0, bits1]
        # H_ind = np.random.randint(0, len(H_stack))
        HH = H_stack[H_ind]
        mode = 0
        SNRdb = 25
        Pilotnum = 32
        # mode = np.random.choice(modes, 1)[0]
        # SNRdb = np.random.choice(SNRdbs, 1)[0]
        # Pilotnum = np.random.choice(Pilotnums, 1)[0]
        YY = MIMO(X, HH, SNRdb, mode, Pilotnum) / 20
        YY = np.reshape(YY, [2, 2, 2, 256], order = 'F')
        XX = np.concatenate((bits0, bits1), 0)
        input_labels.append(XX)
        input_samples.append(YY)
    batch_y = torch.Tensor(np.asarray(input_samples)).to(device)
    batch_x = torch.Tensor(np.asarray(input_labels)).to(device)
    return (batch_y, batch_x)


# if __name__ == '__main__':
#     ###########################以下仅为信道数据载入和链路使用范例############

#     data1=open('/data/CuiMingyao/AI_competition/OFDMReceiver/H.bin','rb')
#     H1=struct.unpack('f'*2*2*2*32*320000, data1.read(4*2*2*2*32*320000))
#     H1=np.reshape(H1,[320000,2,4,32])
#     H=H1[:,1,:,:]+1j*H1[:,0,:,:]

#     Htest=H[300000:,:,:]
#     Htrain=H[:300000,:,:]

#     batch = 2
#     Y, X, H = generatorXY(batch, Htrain)
    
#     # np.savetxt('./Y_1.csv', Y, delimiter=',')
#     # X_1 = np.array(np.floor(X + 0.5), dtype=np.bool)
#     # X_1.tofile('./X_1.bin')


#     #####################数据维度说明####################

#     ### 接受信号维度说明
#     ### 这里一定要加order！！！！######
#     # 样本数 * 虚实部 * 导频and数据 * 天线数 * 子载波数
#     output = np.reshape(Y, (-1, 2,2,2, 256), order = 'F') 

#     ###例：提取复数数据
#     # 输出维度为 样本数 * 天线数 * 子载波数
#     output_complex = output[:,0,:,:,:] + 1j*output[:,1,:,:,:]
#     output_data = output_complex[:,1,:,:]


#     ### 信道维度说明，这里一定要加order！！！！！
#     # H:时域信道，样本数 * 天线数(4，依次为第一根Tx至第一根Rx“00”、“01”、“10”、“11” ) * 时域维度(32)
#     Hf = np.fft.fft(H, 256)/20
#     # Hf：频域信道，样本数 * Rx * Tx * 频域维度
#     # Rx * Tx 为 2*2 的矩阵 从上至下，从左至右依次为第一根Tx至第一根Rx、第一根Tx至第二根Rx、第二根Tx至第一根Rx，第二根Tx至第二根Rx
#     # 因为MIMO接受数据有除以20，这里就除以了20
#     Hf = np.reshape(Hf, (-1, 2,2,256), order='F')



#     #### 输入数据维度说明，这里一定【不要】加order！！！！！！
#     # X：样本数 * 发射天线数 * （子载波数 \times 虚实部）
#     X = np.reshape(X, (-1, 2,512))

#     # 可用的复数input：样本数 * 发射天线数 * 子载波数
#     input = np.zeros((batch, 2,256), dtype=complex)
#     for num in range(batch):
#         for idx in range(2):
#             bits = X[num, idx, :]
#             # 与 utils文件里的modulation一模一样的操作
#             bits = np.reshape(bits, (256,2))
#             input[num, idx,:] = 0.7071 * (2 * bits[:, 0] - 1) + 0.7071j * (2 * bits[:, 1] - 1)

#     # 测试是否满足 Y - HX = 0
#     # 各数据维度总结【均为复数】
#     # input： batch * 2 * 256    【 样本数 * 发射天线数 * 子载波数 】  
#     # Hf：    batch * 2 * 2 *256 【 样本数 * 收端天线数 * 发端天线数 *子载波数 】  
#     # output_data 或 Y_hat：batch * 2 * 256 【 样本数 * 收端天线数 * 子载波数 】  
#     Y_hat = np.zeros(output_data.shape, dtype=complex)
#     for num in range(batch):
#         for idx in range(256):
#             Y_hat[num, :,idx] = Hf[num, :,:,idx].dot(input[num, :,idx])

#     error = output_data - Y_hat
#     print('Hello world')

