from utils import *
import struct
import torch
from Model_define_pytorch import *

from MLreceiver import MLReceiver, MakeCodebook
from generate_data import generatorXY

# Parameters for training
gpu_list = '6,7'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

data1=open('H.bin','rb')
H1=struct.unpack('f'*2*2*2*32*320000,data1.read(4*2*2*2*32*320000))
H1=np.reshape(H1,[320000,2,4,32])
H_tra = H1[:,1,:,:]+1j*H1[:,0,:,:]

data2 = open('H_val.bin','rb')
H2 = struct.unpack('f'*2*2*2*32*2000,data2.read(4*2*2*2*32*2000))
H2 = np.reshape(H2,[2000,2,4,32])
H_val = H2[:,1,:,:]+1j*H2[:,0,:,:]

Pilot_num = 32
batch_num = 2000
Ns = batch_num
SNRdb = -1
mode = 0
act='ELU'

 # 生成测试数据 Y：-1*2048； X：-1*1024； H：-1*4*32 时域信道
Y, X, H = generatorXY(batch_num,H_val,Pilot_num, SNRdb=-1, mode= 0)

# data_load_address = '/data/AI_Wireless/'
# Y = np.load(data_load_address+'validation_Y_P='+str(Pilot_num)+'_mode='+str(mode)+'_SNR='+str(SNRdb)+'.npy')
# Ns = Y.shape[0]
# H = H_val

##### Obtain input data #####
Y = np.reshape(Y, [-1, 2, 2, 2, 256], order='F').astype(np.float32)
Yp = Y[:,:,0,:,:]   # 取出接收导频信号，实部虚部*两根接收天线*256子载波
Yd = Y[:,:,1,:,:]   # 取出接收导频信号，实部虚部*两根接收天线*256子载波

##### 根据不同mode估计时域信道 #####
in_dim = 1024
h_dim = 4096
out_dim = 2048
n_blocks = 2

# Model Construction #input: batch*2*2*256 Received Yp # output: batch*2*2*256 time-domain channel
FC = yp2h_estimation(in_dim, h_dim, out_dim, n_blocks)

FC = torch.nn.DataParallel(FC).cuda()  # model.module
# Load weights
FC_path = '/home/wxh/AI_wireless_communication/Modelsave/FC_CE_Pilot'+ str(Pilot_num)+'_mode' + str(mode) + '.pth.tar'
FC.load_state_dict(torch.load(FC_path)['state_dict'])
print("Model for FC CE has been loaded!")
# Estimate the time-domain channel
FC.eval()
Hf_output1 = FC(torch.tensor(Yp.reshape(batch_num, 2*2*256)).to('cuda'))

CNN = CNN_Estimation()
CNN = torch.nn.DataParallel( CNN ).cuda()  # model.module
CNN_path = '/home/wxh/AI_wireless_communication/Modelsave/CNN_CE_Pilot'+ str(Pilot_num)+'_mode' + str(mode) + '.pth.tar'
CNN.load_state_dict(torch.load( CNN_path )['state_dict'])
print("Model for CNN CE has been loaded!")
CNN.eval()

Hf_output1 = Hf_output1.reshape(batch_num, 2, 4, 256)
input = torch.cat([torch.tensor(Yd).cuda(), Hf_output1], 2)
Ht_output = CNN(input)
Ht_output = Ht_output.reshape(batch_num, 2, 4, 32)

Ht_output1 = Ht_output.cuda().data.cpu().numpy()
Ht_hat = Ht_output1[:,0,:,:]+1j*Ht_output1[:,1,:,:]

Hf_hat = np.fft.fft(Ht_hat, 256) / 20  # frequency-domain channel
Hf_hat = np.reshape(Hf_hat, [Ns, 2, 2, 256], order='F')

Hf = np.fft.fft(H, 256) / 20  # frequency-domain channel
Hf = np.reshape(Hf, [Ns, 2, 2, 256], order='F')


NMSE = np.mean(abs(Hf_hat-Hf)**2)/np.mean(abs(Hf)**2)
print('NMSE: ' +str(NMSE))


#### ML的输入 ####
Yc = Y [:,0,:,:,:] + 1j*Y [:,1,:,:,:]
Yd = Yc[:,1,:,:]

# 生成码本
G = 4
codebook = MakeCodebook(G)

#### 基于ML恢复发送比特流X ####

# X_ML1, X_bits1 = MLReceiver(Y,Hf,codebook)
X_ML2, X_bits2 = MLReceiver(Yd,Hf_hat,codebook)

# 可用的复数input：样本数 * 发射天线数 * 子载波数
X1 = np.reshape(X, (-1, 2, 512))
input = np.zeros((batch_num, 2,256), dtype=complex)
for num in range(batch_num):
    for idx in range(2):
        bits = X1[num, idx, :]
        # 与 utils文件里的modulation一模一样的操作
        bits = np.reshape(bits, (256,2))
        input[num, idx,:] = 0.7071 * (2 * bits[:, 0] - 1) + 0.7071j * (2 * bits[:, 1] - 1)


error2 = X_bits2 - X
bit_error2 = np.sum(np.sum(np.abs(error2)<0.1))/ error2.size
print('Accuracy: ' +str(bit_error2))


#***************基于resnet18网络的信号检测*************************
model_for_detection = ResNet()
model_for_detection = torch.nn.DataParallel(model_for_detection).cuda()  # model.module
# Load weights
model_path = '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/Resnet18_SD_mode0.pth.tar'
model_for_detection.load_state_dict(torch.load(model_path)['state_dict'])
print(model_path + ' has been loaded!')
print("Model for CNN SD has been loaded!")
model_for_detection.eval()
Yd = Y[:, :, 1, :, :]  # 取出接收数据信号，实部虚部*2*256

Hf_hat = np.reshape(Hf_hat, [Ns, 4, 256], order='F')
Hf_input = torch.zeros([Ns, 2, 4, 256], dtype=torch.float32)
Hf_input[:, 0, :, :] = torch.tensor(Hf_hat.real, dtype=torch.float32)
Hf_input[:, 1, :, :] = torch.tensor(Hf_hat.imag, dtype=torch.float32)

input = torch.cat([torch.tensor(Yd), Hf_input], 2)
input = torch.reshape(input, [Ns, 1, 12, 256])
input = input.cuda()
output = model_for_detection(input)

label = X
output = (output > 0.5)*1.
output = output.cpu().detach().numpy()

eps = 0.1
error = np.abs(output - label)
average_accuracy = np.sum(error < eps) / error.size
print('Accuracy: ' +str(average_accuracy))
