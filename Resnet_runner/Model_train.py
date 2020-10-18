## -*- coding: utf-8 -*-
'''
seefun . Aug 2020.
github.com/seefun | kaggle.com/seefun
'''

import numpy as np
import h5py
import torch
import os
import torch.nn as nn
import random
import time
import argparse
from torch.optim import lr_scheduler
from Model_define_pytorch import AutoEncoder, DatasetFolder, NMSE_cuda, NMSELoss


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# fozu



# time
timestr = time.strftime("%m_%d_%H:%M", time.localtime())

# Parameters for training
parser = argparse.ArgumentParser()


parser.add_argument('--gpu_list',  type = str,  default='6', help='input gpu list')
parser.add_argument('--data_path',  type= str,  default = '/data/CuiMingyao/AI_competition/compression/H_train.mat')

parser.add_argument('--seed',  type= int,  default=42, help='input random seed')
parser.add_argument('--batch_size',  type= int,  default=256)
parser.add_argument('--epochs',  type= int,  default=300)
parser.add_argument('--lr',  type= float,  default = 2e-3)

# freeze 
parser.add_argument('--freeze_encoder',  type= bool,  default = False)
parser.add_argument('--freeze_decoder',  type= bool,  default = False)

parser.add_argument('--num_workers',  type= int,  default = 4)
parser.add_argument('--DirectQuantization',  type= bool,  default = False)

parser.add_argument('--load_encoder',  type= bool,  default = False)
parser.add_argument('--encoder_path',  type= str,  default = './Modelsave/encoder.pth.tar')

parser.add_argument('--load_decoder',  type= bool,  default = False)
parser.add_argument('--decoder_path',  type= str,  default = './Modelsave/decoder.pth.tar')




args = parser.parse_args()



print('GPU LIST:', args.gpu_list)
gpu_list = args.gpu_list
batch_size = args.batch_size
SEED = args.seed
epochs = args.epochs
num_workers = args.num_workers
load_encoder = args.load_encoder
load_decoder = args.load_decoder

DirectQuantization = args.DirectQuantization

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

seed_everything(SEED) 



epochs_qutization = epochs//10

learning_rate = args.lr # bigger to train faster

print_freq = 500  
train_test_ratio = 0.8
# parameters for data
feedback_bits = 128
img_height = 16
img_width = 32
img_channels = 2



# Model construction
model = AutoEncoder(feedback_bits)

if load_encoder:
    model_path = args.encoder_path
    model.encoder.load_state_dict(torch.load(model_path)['state_dict'])
    print("encoder Weight Loaded!")

if load_decoder:
    model_path = args.decoder_path
    model.decoder.load_state_dict(torch.load(model_path)['state_dict'])
    print("decoder Weight Loaded!")

if args.freeze_encoder:
    for params in model.encoder.parameters():
        params.requires_grad = False
    print('freeze encoder!')

if args.freeze_decoder:
    for params in model.decoder.parameters():
        params.requires_grad = False
    print('freeze decoder!')


if len(gpu_list.split(',')) > 1:
    model = torch.nn.DataParallel(model).cuda()  # model.module
else:
    model = model.cuda()



criterion = NMSELoss(reduction='mean') #nn.MSELoss()
criterion_test = NMSELoss(reduction='sum')


    

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

scheduler = lr_scheduler.StepLR(optimizer, step_size = 20, gamma=0.25)


# Data loading
# data_load_address = '../data'
mat = h5py.File(args.data_path, 'r')
data = np.transpose(mat['H_train'])  # shape=(320000, 1024)
data = data.astype('float32')
data = np.reshape(data, [len(data), img_channels, img_height, img_width])
# split data for training(80%) and validation(20%)
np.random.shuffle(data)
start = int(data.shape[0] * train_test_ratio)
x_train, x_test = data[:start], data[start:]

# dataLoader for training
train_dataset = DatasetFolder(x_train)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)

# dataLoader for training
test_dataset = DatasetFolder(x_test)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)



best_loss = 100
for epoch in range(epochs):
    print('========================')
    print('lr:%.4e'%optimizer.param_groups[0]['lr']) 
    # model training
    model.train()
    if epoch < epochs_qutization and not DirectQuantization:
        try:
            model.encoder.quantization = False
            model.decoder.quantization = False
        except:
            model.module.encoder.quantization = False
            model.module.decoder.quantization = False
    else:
        print('*****Run Quantization******')
        try:
            model.encoder.quantization = True
            model.decoder.quantization = True
        except:
            model.module.encoder.quantization = True
            model.module.decoder.quantization = True
        
    #if epoch == epochs//4 * 3:
    #    optimizer.param_groups[0]['lr'] =  optimizer.param_groups[0]['lr'] * 0.25

    
    
    for i, input in enumerate(train_loader):
            
        input = input.cuda()
        output = model(input)
        
        loss = criterion(output, input)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if i % print_freq == 0:
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'Loss: {loss:.6f}\t'.format(
                epoch, epochs, i, len(train_loader), loss=loss.item()) ,time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    model.eval()
    try:
        model.encoder.quantization = True
        model.decoder.quantization = True
    except:
        model.module.encoder.quantization = True
        model.module.decoder.quantization = True
    total_loss = 0
    scheduler.step()
    
    
    with torch.no_grad():
        for i, input in enumerate(test_loader):
            # convert numpy to Tensor
            input = input.cuda()
            output = model(input)
            total_loss += criterion_test(output, input).item()
        average_loss = total_loss / len(test_dataset)
        print('-'*50)
        print('NMSE %.4f'%average_loss)
        print('-'*50)
        if average_loss < best_loss:
            # model save
            model_save = './Modelsave/'
            path1 = 'encoder_'+ timestr +'.pth.tar'
            path2 = 'decoder_'+ timestr +'.pth.tar'
            
            modelSave1 = model_save + path1

            try:
                torch.save({'state_dict': model.encoder.state_dict(), }, modelSave1, _use_new_zipfile_serialization=False)
            except:
                torch.save({'state_dict': model.module.encoder.state_dict(), }, modelSave1, _use_new_zipfile_serialization=False)
            # save decoder
            modelSave2 = model_save + path2
            try:
                torch.save({'state_dict': model.decoder.state_dict(), }, modelSave2, _use_new_zipfile_serialization=False)
            except:
                torch.save({'state_dict': model.module.decoder.state_dict(), }, modelSave2,_use_new_zipfile_serialization=False)
            print('Model saved!')
            best_loss = average_loss

