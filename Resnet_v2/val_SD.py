from utils import *
import struct
import torch
from Model_define_pytorch import *
import random
from MLreceiver import MLReceiver, MakeCodebook
from generate_data import *
from torch.utils.data import Dataset, DataLoader
import os 
# Parameters for training
gpu_list = '4,5,6,7'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list


data2 = open('/data/CuiMingyao/AI_competition/OFDMReceiver/H_val.bin','rb')
H2 = struct.unpack('f'*2*2*2*32*2000,data2.read(4*2*2*2*32*2000))
H2 = np.reshape(H2,[2000,2,4,32])
H_val = H2[:,1,:,:]+1j*H2[:,0,:,:]

Pilot_num = 8
batch_num = 256
Ns = batch_num
SNRdb = -1
mode = 0


test_dataset  = RandomDataset(H_val,Pilot_num=Pilot_num,SNRdb=SNRdb,mode=mode)
test_dataloader = DataLoader(dataset = test_dataset, batch_size = 200, shuffle = True, num_workers = 16, drop_last = True, pin_memory = True)

model_for_detection = ResNet18()
model_for_detection = torch.nn.DataParallel(model_for_detection).cuda()  # model.module
# Load weights
model_path = '/data/CuiMingyao/AI_competition/OFDMReceiver/Modelsave/Resnet18_SD_mode0.pth.tar'
model_for_detection.load_state_dict(torch.load(model_path)['state_dict'])
print(model_path + ' has been loaded!')
print("Weight Loaded!")
model_for_detection.eval()

for Y, X, H in test_dataloader:
    
    Ns = Y.shape[0]

    ##### Obtain input data and output data#####
    Y_input = np.reshape(Y, [Ns, 2, 2, 2, 256], order='F')
    Y_input = Y_input.float()
    Y_input = Y_input[:,:,1,:,:]

    Hf = np.fft.fft(np.array(H), 256) / 20  # perfect frequency-domain channel

    Hf_input = torch.zeros([Ns, 2, 4, 256], dtype=torch.float32)
    Hf_input[:, 0, :, :] = torch.tensor(Hf.real, dtype=torch.float32)
    Hf_input[:, 1, :, :] = torch.tensor(Hf.imag, dtype=torch.float32)

    input = torch.cat([Y_input, Hf_input], 2)
    input = torch.reshape(input, [Ns, 1, 12, 256])
    input = input.cuda()
    output = model_for_detection(input)

    label = X.float().numpy()
    output = (output > 0.5)*1.
    output = output.cpu().detach().numpy()

    eps = 0.1
    error = np.abs(output - label)
    average_accuracy = np.sum(error < eps) / error.size
    print(average_accuracy)