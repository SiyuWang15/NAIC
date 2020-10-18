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
import logging

from Model_define_pytorch import AutoEncoder, DatasetFolder, NMSE_cuda, NMSELoss

# Parameters for training
gpu_list = '0,1,2,3'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
SEED = 42
seed_everything(SEED) 

batch_size = 256 * 4
log = 'training.log'
epochs = 300
epochs_qutization = 10

learning_rate = 2e-3 # bigger to train faster
num_workers = 0
print_freq = 500  
train_test_ratio = 0.8
# parameters for data
feedback_bits = 128
img_height = 16
img_width = 32
img_channels = 2


load_model = False
DirectQuantization = False
# Model construction
model = AutoEncoder(feedback_bits)

if load_model:
    model_path = './Modelsave/encoder.pth.tar'
    model.encoder.load_state_dict(torch.load(model_path)['state_dict'])
    
    model_path = './Modelsave/decoder.pth.tar'
    model.decoder.load_state_dict(torch.load(model_path)['state_dict'])
    print("weight loaded")

# logging
level = getattr(logging, 'INFO', None)
if not isinstance(level, int):
    raise ValueError('level {} not supported'.format(args.verbose))
handler1 = logging.StreamHandler()
if os.path.exists(log):
    os.remove(log)
handler2 = logging.FileHandler(log)
formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
handler1.setFormatter(formatter)
handler2.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(handler1)
logger.addHandler(handler2)
logger.setLevel(level)


logging.info('batch_size: {}, log: {}'.format(batch_size, log))


model.encoder.quantization = False
model.decoder.quantization = False

if len(gpu_list.split(',')) > 1:
    model = torch.nn.DataParallel(model).cuda()  # model.module
else:
    model = model.cuda()

criterion = NMSELoss(reduction='mean') #nn.MSELoss()
criterion_test = NMSELoss(reduction='sum')

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Data loading
data_load_address = './data'
mat = h5py.File(data_load_address + '/H_train.mat', 'r')
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
    logging.info('lr:%.4e'%optimizer.param_groups[0]['lr']) 
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
        print('love live!')
        try:
            model.encoder.quantization = True
            model.decoder.quantization = True
        except:
            model.module.encoder.quantization = True
            model.module.decoder.quantization = True
        
    if epoch == epochs//4 * 3:
        optimizer.param_groups[0]['lr'] =  optimizer.param_groups[0]['lr'] * 0.25
    
    for i, input in enumerate(train_loader):
            
        input = input.cuda()
        output = model(input)
        
        loss = criterion(output, input)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if i % print_freq == 0:
            logging.info('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss:.4f}\t'.format(
                epoch, i, len(train_loader), loss=loss.item()))

    model.eval()
    try:
        model.encoder.quantization = True
        model.decoder.quantization = True
    except:
        model.module.encoder.quantization = True
        model.module.decoder.quantization = True
    total_loss = 0
    with torch.no_grad():
        for i, input in enumerate(test_loader):
            # convert numpy to Tensor
            input = input.cuda()
            output = model(input)
            total_loss += criterion_test(output, input).item()
        average_loss = total_loss / len(test_dataset)
        logging.info('-'*50)
        logging.info('NMSE %.4f'%average_loss)
        logging.info('-'*50)
        if average_loss < best_loss:
            # model save
            model_save = './Modelsave/'
            path1 = 'encoder_NMSE_%.4f.pth.tar'%average_loss
            path2 = 'decoder_NMSE_%.4f.pth.tar'%average_loss
            
            modelSave1 = model_save + path1

            try:
                torch.save({'state_dict': model.encoder.state_dict(), }, modelSave1)
            except:
                torch.save({'state_dict': model.module.encoder.state_dict(), }, modelSave1)
            # save decoder
            modelSave2 = model_save + path2
            try:
                torch.save({'state_dict': model.decoder.state_dict(), }, modelSave2)
            except:
                torch.save({'state_dict': model.module.decoder.state_dict(), }, modelSave2)
            logging.info('Model saved!')
            best_loss = average_loss

