import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np 
import logging
import os
import sys
import copy
sys.path.append('..')
from Estimators import CNN_Estimation, FC_ELU_Estimation, NMSELoss, ResNet34, ResNet50, ResNet101, Densenet, XYH2X_ResNet18, XDH2H_Resnet
from data import get_YH_data_random
from utils import *

class Y2HRunner():
    def __init__(self, config, cnn:bool=False):
        self.config = config
        self.mode = config.mode
        self.Pn = config.Pn
    
    def get_optimizer(self, parameters, lr):
        if self.config.train.optimizer == 'adam':
            return optim.Adam(parameters, lr = lr)
        elif self.config.train.optimizer == 'sgd':
            return optim.SGD(parameters, lr = lr, momentum=0.9)
        else:
            raise NotImplementedError('Optimizer {} not understood.'.format(self.config.train.optimizer))

    def run(self):
        if self.config.model == 'fc':
            self.train_FC()
        elif self.config.model == 'cnn':
            self.train_CNN()
    
    def get_dataloader(self):
        train_set, val_set = get_YH_data_random(self.mode, self.Pn)
        train_loader = DataLoader(train_set, batch_size=self.config.train.train_batch_size, \
            shuffle = True, num_workers = 16, drop_last = False)
        val_loader = DataLoader(val_set, batch_size=self.config.train.val_batch_size, \
            shuffle=False, num_workers=16, drop_last=False)
        return train_loader, val_loader
    
    def LSequalization(self, h, y):
        # y 复数： batch*2*256
        # h 复数： batch*2*2*256
        batch = h.shape[0]
        norm = h[:,0,0,:] * h[:,1,1,:] - h[:,0,1,:] * h[:,1,0,:]
        norm = norm.reshape([batch,1,1,256])
        invh = np.zeros([batch,2,2,256], dtype = np.complex64)
        invh[:,0,0,:] = h[:,1,1,:]
        invh[:,1,1,:] = h[:,0,0,:]
        invh[:,0,1,:] = -h[:,0,1,:]
        invh[:,1,0,:] = -h[:,1,0,:]
        invh = invh/norm
        y = y.reshape(batch, 1, 2, 256)
        
        x_LS = invh*y
        x_LS = x_LS[:,:,0,:] + x_LS[:,:,1,:]
        return x_LS

    def get_LS(self, Yd_input, Hf):
        Yd = np.array(Yd_input[:,0,:,:] + 1j*Yd_input[:,1,:,:])
        Hf = np.array(Hf[:,0,:,:] + 1j*Hf[:,1,:,:]) 
        Hf = np.reshape(Hf, [-1,2,2,256], order = 'F')
        X_LS = self.LSequalization(Hf, Yd)
        X_LS.real = (X_LS.real > 0)*2 - 1
        X_LS.imag = (X_LS.imag > 0)*2 - 1
        return X_LS

    def train_FC(self):
        device = 'cuda'
        train_loader, val_loader = self.get_dataloader()
        logging.info('Data Loaded!')
        FC = FC_ELU_Estimation(self.config.FC.in_dim, self.config.FC.h_dim, self.config.FC.out_dim, self.config.FC.n_blocks)
        if not self.config.train.FC_resume == 'None':
            fp = os.path.join(f'/data/siyu/NAIC/workspace/ResnetY2HEstimator/mode_{self.mode}_Pn_{self.Pn}/FC', self.config.train.FC_resume, 'checkpoints/best.pth')
            try:
                FC.load_state_dict(torch.load(fp))
            except:
                FC.load_state_dict(torch.load(fp)['fp'])
            logging.info(f'Loading state dict of FC from {self.config.train.FC_resume}')
        FC.to(device)

        criterion = NMSELoss()
        optimizer = self.get_optimizer(FC.parameters(), self.config.train.fc_lr)
        best_nmse = 1000.

        logging.info('Everything prepared well, start to train...')

        for epoch in range(self.config.n_epochs):
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f'Epoch [{epoch}]/[{self.config.n_epochs}] learning rate: {current_lr:.4e}')

            FC.train()
            for it, (Yp4fc, _, _, _, H_label) in enumerate(train_loader):
                bs = len(Yp4fc)
                optimizer.zero_grad()
                # 真实的频域信道，获取标签
                Hf_label = process_H(H_label).to(device)
                # 第一层网络输入
                Yp4fc = Yp4fc.to(device)
                Hf = FC(Yp4fc).reshape(bs, 2, 4, 256)
                # 计算loss
                loss = criterion(Hf, Hf_label)
                loss.backward()
                optimizer.step()
                if it % self.config.print_freq == 0:
                    # print(nmse)
                    logging.info(f'Mode:{self.mode} || Epoch: [{epoch}/{self.config.n_epochs}][{it}/{len(train_loader)}]\t Loss {loss.item():.5f}')
            FC.eval()
            with torch.no_grad():
                Hf_list = []
                Hflabel_list = []
                for Yp4fc, _, _, _, H_label in val_loader:
                    bs = Yp4fc.shape[0]
                    # 真实的频域信道，获取标签
                    H_label = process_H(H_label)

                    Yp4fc = Yp4fc.to(device)
                    Hf = FC(Yp4fc).reshape(bs, 2, 4, 256).cpu()
                    Hf_list.append(Hf)
                    Hflabel_list.append(H_label)
                Hf = torch.cat(Hf_list, dim = 0)
                Hflabel = torch.cat(Hflabel_list, dim = 0)
                loss = criterion(Hf, Hflabel)

                fp = os.path.join(self.config.ckpt_dir, f'epoch{epoch}.pth')
                torch.save({'fc': FC.state_dict()}, fp)
                if loss < best_nmse:
                    torch.save({'fc': FC.state_dict()}, os.path.join(self.config.ckpt_dir, 'best.pth'))
                    best_nmse = loss.item()
                logging.info(f'{fp} saved!')
                logging.info(f'Epoch [{epoch}]/[{self.config.n_epochs}] || NMSE {loss.item():.5f}, best nmse: {best_nmse:.5f}')

    def train_CNN(self):
        device = 'cuda'
        train_set, val_set = get_YH_data_random(self.mode, self.Pn)
        train_loader, val_loader = self.get_dataloader()
        logging.info('Data Loaded!')

        FC = FC_ELU_Estimation(self.config.FC.in_dim, self.config.FC.h_dim, self.config.FC.out_dim, self.config.FC.n_blocks)
        FC.to(device)
        if self.config.cnnmodel == 'base':
            CNN = CNN_Estimation().to(device)
        elif self.config.cnnmodel == 'resnet34':
            CNN = ResNet34().to(device)
        elif self.config.cnnmodel == 'resnet50':
            CNN = ResNet50().to(device)
        elif self.config.cnnmodel == 'resnet101':
            CNN = nn.DataParallel(ResNet101()).to(device)
        elif self.config.cnnmodel == 'densenet':
            CNN = nn.DataParallel(Densenet()).to(device)
            logging.info('This is densenet.')
        if not self.config.train.CNN_resume == 'None':
            fp = os.path.join(f'/data/siyu/NAIC/workspace/ResnetY2HEstimator/mode_{self.mode}_Pn_{self.Pn}/CNN', \
                self.config.train.CNN_resume, 'checkpoints/best.pth')
            state_dicts = torch.load(fp)
            CNN.load_state_dict(state_dicts['cnn'])
            FC.load_state_dict(state_dicts['fc'])
            logging.info(f'load state dicts of CNN and FC from {fp}.')
        else:
            assert not self.config.train.FC_resume == 'None'
            fp = os.path.join(f'/data/siyu/NAIC/workspace/ResnetY2HEstimator/mode_{self.mode}_Pn_{self.Pn}/FC', self.config.train.FC_resume, 'checkpoints/best.pth')
            try:
                FC.load_state_dict(torch.load(fp)['fc'])
            except:
                FC.load_state_dict(torch.load(fp))
            logging.info(f'Loading state dict of FC from {self.config.train.FC_resume}, random initialize CNN')
        
        if self.config.train.freeze_FC:
            for param in FC.parameters():
                param.requires_grad = False
            logging.info('Freeze FC layer.')
            FC.eval()
        
        best_nmse = 1000.
        
        for param in CNN.parameters():
            param.requires_grad = False
        CNN.eval()
        logging.info('freeze CNN.')
        
        SD = XYH2X_ResNet18()
        fp = '/data/siyu/NAIC/workspace/mingyao/XYH2X_Resnet18_SD_mode0_Pilot8.pth.tar'
        SD.load_state_dict(torch.load(fp)['state_dict'])
        for param in SD.parameters():
            param.requires_grad = False
        SD.eval()
        SD.to(device)
        logging.info(f'load state dict of SD and freeze it.')

        CE2 = XDH2H_Resnet()
        fp = '/data/siyu/NAIC/workspace/mingyao/XDH2H_Resnet34_SD_mode0_Pilot8.pth.tar'
        CE2.load_state_dict(torch.load(fp)['state_dict'])
        CE3 = copy.deepcopy(CE2)
        CE2.eval()
        CE2.to(device)
        CE3.to(device)
        for param in CE2.parameters():
            param.requires_grad = False
        logging.info('freeze CE2.')

        FC = nn.DataParallel(FC).to(device)
        CNN = nn.DataParallel(CNN).to(device)
        SD = nn.DataParallel(SD).to(device)
        CE2 = nn.DataParallel(CE2).to(device)
        CE3 = nn.DataParallel(CE3).to(device)

        optimizer_CE3 = self.get_optimizer(CE3.module.parameters(), lr = 0.0001)
        CE3.train()

        criterion = NMSELoss()
        # optimizer_CNN = self.get_optimizer(filter(lambda p: p.requires_grad,  CNN.parameters()), self.config.train.cnn_lr)
         
        logging.info('Everything prepared well, start to train...')
        for epoch in range(self.config.n_epochs):
            CE3.eval()
            with torch.no_grad():
                Ht1_list = []
                Ht2_list = []
                Ht3_list = []
                Hlabel_list = []
                for Yp4fc, Yp4cnn, Yd, X, H_label in val_loader:
                    bs = Yp4fc.shape[0]
                    Yp4fc = Yp4fc.to(device)
                    Hf = FC(Yp4fc)
                    Yp4fc = Yp4fc.cpu() # release gpu memory
                    Hf = Hf.reshape(bs, 2, 4, 256)
                    cnn_input = torch.cat([Yd.to(device), Yp4cnn.to(device), Hf], dim = 2)
                    output2 = CNN(cnn_input).reshape(bs, 2, 4, 32)
                    
                    H_test_padding = copy.deepcopy(output2)
                    output2 = output2.cpu() 
                    H_test_padding = torch.cat([H_test_padding, torch.zeros(bs,2,4,256-32, requires_grad=True).to(device)],3)
                    H_test_padding = H_test_padding.permute(0,2,3,1)

                    H_test_padding = torch.fft(H_test_padding, 1)/20
                    H_test_padding = H_test_padding.permute(0,3,1,2).contiguous() 

                    X_LS = self.get_LS(Yd, H_test_padding.detach().cpu())
                    X_input_test = torch.zeros([bs, 2, 2, 256], dtype = torch.float32)
                    X_input_test[:,0,:,:] = torch.tensor(X_LS.real, dtype = torch.float32)
                    X_input_test[:,1,:,:] = torch.tensor(X_LS.imag, dtype = torch.float32)

                    input3 = torch.cat([X_input_test.cuda(), Yp4cnn.to(device), Yd.cuda(), H_test_padding], 2)
                    output3 = SD(input3)

                    output3 = output3.reshape([bs,2,256,2])
                    output3 = output3.permute(0,3,1,2).contiguous()

                    # input4 = torch.cat([X_input_test.cuda(), Yd_input_test.cuda(), H_test_padding], 2)
                    input4 = torch.cat([output3.cuda(), Yd.cuda(), H_test_padding], dim=2)

                    output4 = CE2(input4).reshape(bs, 2, 4, 32)

                    H_test_padding = copy.deepcopy(output4)
                    output4 = output4.cpu()

                    H_test_padding = torch.cat([H_test_padding, torch.zeros(bs,2,4,256-32, requires_grad=True).to(device)],dim=3)
                    H_test_padding = H_test_padding.permute(0,2,3,1)

                    H_test_padding = torch.fft(H_test_padding, 1)/20
                    H_test_padding = H_test_padding.permute(0,3,1,2).contiguous() 

                    X_LS = self.get_LS(Yd, H_test_padding.detach().cpu())
                    X_input_test = torch.zeros([bs, 2, 2, 256], dtype = torch.float32)
                    X_input_test[:,0,:,:] = torch.tensor(X_LS.real, dtype = torch.float32)
                    X_input_test[:,1,:,:] = torch.tensor(X_LS.imag, dtype = torch.float32)
                    
                    input5 = torch.cat([X_input_test.to(device), Yp4cnn.to(device), Yd.to(device), H_test_padding], dim = 2)
                    output5 = SD(input5).reshape([bs, 2, 256, 2])
                    output5 = output5.permute(0, 3, 1, 2).contiguous()

                    input6 = torch.cat([output5, Yd.cuda(), H_test_padding], 2)
                    output6 =  CE3(input6).reshape(bs, 2, 4, 32).cpu()


                    # 计算loss
                    # nmse1 = criterion(output2.reshape(bs, 2, 4, 32).detach().cpu(), Ht_test_label)
                    # nmse1 = nmse1.numpy()
                    # nmse2 = criterion(output4.detach().cpu(), Ht_test_label)
                    # nmse2 = nmse2.numpy()


                    Ht1_list.append(output2)
                    Ht2_list.append(output4)
                    Ht3_list.append(output6)
                    Hlabel_list.append(H_label)

                Ht1 = torch.cat(Ht1_list, dim = 0)
                Ht2 = torch.cat(Ht2_list, dim = 0)
                Ht3 = torch.cat(Ht3_list, dim = 0)
                Hlabel = torch.cat(Hlabel_list, dim = 0)
                loss1, loss2, loss3 = criterion(Ht1, Hlabel), criterion(Ht2, Hlabel), criterion(Ht3, Hlabel)
                if loss3 < best_nmse:
                    best_nmse = loss3.item()
                    state_dicts = {
                        'CE3': CE3.state_dict(),
                        'epoch_num': epoch
                    }
                    torch.save(state_dicts, os.path.join(self.config.ckpt_dir, f'best.pth'))
                if epoch % self.config.save_freq == 0:
                    fp = os.path.join(self.config.ckpt_dir, f'epoch{epoch}.pth')
                    state_dicts = {
                        'CE3': CE3.state_dict()
                    }
                    torch.save(state_dicts, fp)
                    logging.info(f'{fp} saved.')
                logging.info(f'Validation Epoch [{epoch}]/[{self.config.n_epochs}] || nmse1 {loss1.item():.5f}, nmse2: {loss2.item():.5f}, nmse3: {loss3.item():.5f}, best nmse: {best_nmse:.5f}')
            
            CE3.train()
            for it, (Yp4fc, Yp4cnn, Yd, X, H_label) in enumerate(train_loader): # H_train: bsx2(real and imag)x4x32
                bs = Yp4fc.shape[0]
                Yp4fc = Yp4fc.to(device)
                Hf = FC(Yp4fc)
                Yp4fc = Yp4fc.cpu() # release gpu memory
                Hf = Hf.reshape(bs, 2, 4, 256)
                cnn_input = torch.cat([Yd.to(device), Yp4cnn.to(device), Hf], dim = 2)
                output2 = CNN(cnn_input).reshape(bs, 2, 4, 32)
                
                H_test_padding = copy.deepcopy(output2)
                output2 = output2.cpu() 
                H_test_padding = torch.cat([H_test_padding, torch.zeros(bs,2,4,256-32, requires_grad=True).to(device)],3)
                H_test_padding = H_test_padding.permute(0,2,3,1)

                H_test_padding = torch.fft(H_test_padding, 1)/20
                H_test_padding = H_test_padding.permute(0,3,1,2).contiguous() 

                X_LS = self.get_LS(Yd, H_test_padding.detach().cpu())
                X_input_test = torch.zeros([bs, 2, 2, 256], dtype = torch.float32)
                X_input_test[:,0,:,:] = torch.tensor(X_LS.real, dtype = torch.float32)
                X_input_test[:,1,:,:] = torch.tensor(X_LS.imag, dtype = torch.float32)

                input3 = torch.cat([X_input_test.cuda(), Yp4cnn.to(device), Yd.cuda(), H_test_padding], dim = 2)
                output3 = SD(input3)

                output3 = output3.reshape([bs,2,256,2])
                output3 = output3.permute(0,3,1,2).contiguous()

                # input4 = torch.cat([X_input_test.cuda(), Yd_input_test.cuda(), H_test_padding], 2)
                input4 = torch.cat([output3.cuda(), Yd.cuda(), H_test_padding], 2)

                output4 = CE2(input4).reshape(bs, 2, 4, 32)

                Ht2 = output4.cpu()
                H_test_padding = output4


                H_test_padding = torch.cat([H_test_padding, torch.zeros(bs,2,4,256-32, requires_grad=True).to(device)],3)
                H_test_padding = H_test_padding.permute(0,2,3,1)

                H_test_padding = torch.fft(H_test_padding, 1)/20
                H_test_padding = H_test_padding.permute(0,3,1,2).contiguous() 

                X_LS = self.get_LS(Yd, H_test_padding.detach().cpu())
                X_input_test = torch.zeros([bs, 2, 2, 256], dtype = torch.float32)
                X_input_test[:,0,:,:] = torch.tensor(X_LS.real, dtype = torch.float32)
                X_input_test[:,1,:,:] = torch.tensor(X_LS.imag, dtype = torch.float32)
                
                input5 = torch.cat([X_input_test.to(device), Yp4cnn.to(device), Yd.to(device), H_test_padding], dim = 2)
                output5 = SD(input5).reshape([bs, 2, 256, 2])
                output5 = output5.permute(0, 3, 1, 2).contiguous()

                input6 = torch.cat([output5, Yd.cuda(), H_test_padding], 2)
                output6 =  CE3(input6).reshape(bs, 2, 4, 32).cpu()

                loss = criterion(output6, H_label)
                loss.backward()
                optimizer_CE3.step()

                if it % self.config.print_freq == 0:
                    # print(nmse)
                    logging.info(f'CNN Mode:{self.mode} || Epoch: [{epoch}/{self.config.n_epochs}][{it}/{len(train_loader)}]\t Loss {loss.item():.5f}')

            if epoch >0:
                if epoch % self.config.lr_decay ==0:
                    optimizer_CE3.param_groups[0]['lr'] =  max(optimizer_CE3.param_groups[0]['lr'] * 0.5, self.config.lr_threshold)