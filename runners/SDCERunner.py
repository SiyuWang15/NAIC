import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np 
import logging
import os
import copy
from Estimators import CNN_Estimation, FC_ELU_Estimation, NMSELoss, ResNet34, XYH2X_ResNet18, XDH2H_Resnet
from data import get_YH_data_random
from utils import *

class SDCERunner():
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

    def get_dataloader(self):
        train_set, val_set = get_YH_data_random(self.mode, self.Pn)
        train_loader = DataLoader(train_set, batch_size=self.config.train.train_batch_size, \
            shuffle = True, num_workers = 16, drop_last = False)
        val_loader = DataLoader(val_set, batch_size=self.config.train.val_batch_size, \
            shuffle=False, num_workers=16, drop_last=False)
        return train_loader, val_loader

    def run(self):
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
            logging.info("Resnet34.")
        else:
            raise NotImplementedError
        assert not self.config.train.CNN_resume == 'None'
        fp = os.path.join(f'./workspace/ResnetY2HEstimator/mode_{self.mode}_Pn_{self.Pn}/CNN', \
            self.config.train.CNN_resume, 'checkpoints/best.pth')
        state_dicts = torch.load(fp)
        CNN.load_state_dict(state_dicts['cnn'])
        FC.load_state_dict(state_dicts['fc'])
        FC.eval()
        CNN.eval()
        for param in FC.parameters():
            param.requires_grad = False
        for param in CNN.parameters():
            param.requires_grad = False

        logging.info(f'load state dicts of CNN and FC from {fp} and freeze them.')
        
        best_nmse = 1000.

        # state dicts of SD and CE2 are fixed from mingyao's training
        SD = XYH2X_ResNet18()
        fp = './workspace/ResnetY2HEstimator/mode_0_Pn_8/mingyao/XYH2X_Resnet18_SD_mode0_Pilot8.pth.tar' 
        SD.load_state_dict(torch.load(fp)['state_dict'])
        SD.to(device)
        logging.info(f'load state dict of SD and do not freeze it.')

        CE2 = XDH2H_Resnet()
        fp = './workspace/ResnetY2HEstimator/mode_0_Pn_8/mingyao/Best_XDH2H_Resnet34_SD_mode0_Pilot8.pth.tar'
        CE2.load_state_dict(torch.load(fp)['state_dict'])
        CE2.to(device)

        FC = nn.DataParallel(FC).to(device)
        SD = nn.DataParallel(SD).to(device)
        CNN = nn.DataParallel(CNN).to(device)
        CE2 = nn.DataParallel(CE2).to(device)

        optimizer_CE2 = self.get_optimizer(CE2.parameters(), lr = self.config.train.lr)
        optimizer_SD = self.get_optimizer(SD.parameters(), lr = self.config.train.lr)
        criterion = NMSELoss()

        logging.info('Everything prepared well, start to train...')
        for epoch in range(self.config.n_epochs):
            CE2.eval()
            SD.eval()
            with torch.no_grad():
                Ht1_list = []
                Ht2_list = []
                predX = []
                Xlabel_list = []
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
                    predX.append((output3.detach().cpu() > 0.5).float())

                    output3 = output3.reshape([bs,2,256,2])
                    output3 = output3.permute(0,3,1,2).contiguous()

                    # input4 = torch.cat([X_input_test.cuda(), Yd_input_test.cuda(), H_test_padding], 2)
                    input4 = torch.cat([output3.cuda(), Yd.cuda(), H_test_padding], dim=2)

                    output4 = CE2(input4).reshape(bs, 2, 4, 32).detach().cpu()

                    Ht1_list.append(output2)
                    Ht2_list.append(output4)
                    Hlabel_list.append(H_label)
                    Xlabel_list.append(X)

                Ht1 = torch.cat(Ht1_list, dim = 0)
                Ht2 = torch.cat(Ht2_list, dim = 0)
                predX = torch.cat(predX, dim = 0)
                Xlabel = torch.cat(Xlabel_list, dim = 0)
                acc = (predX == Xlabel).float().mean()
                Hlabel = torch.cat(Hlabel_list, dim = 0)
                loss1, loss2 = criterion(Ht1, Hlabel), criterion(Ht2, Hlabel)
                if loss2 < best_nmse:
                    CE2.to('cpu')
                    SD.to('cpu')
                    best_nmse = loss2.item()
                    state_dicts = {
                        'CE2': CE2.state_dict(),
                        'SD': SD.state_dict(), 
                        'epoch_num': epoch
                    }
                    # CE2 = nn.DataParallel(CE2).to(device)
                    # SD = nn.DataParallel(SD).to(device)
                    CE2.to(device)
                    SD.to(device)
                    torch.save(state_dicts, os.path.join(self.config.ckpt_dir, f'best.pth'))
                if epoch % self.config.save_freq == 0:
                    CE2.to('cpu')
                    SD.to('cpu')
                    fp = os.path.join(self.config.ckpt_dir, f'epoch{epoch}.pth')
                    state_dicts = {
                        'CE2': CE2.state_dict(),
                        'SD': SD.state_dict()
                    }
                    # CE2 = nn.DataParallel(CE2).to(device)
                    # SD = nn.DataParallel(SD).to(device)
                    CE2.to(device)
                    SD.to(device)
                    torch.save(state_dicts, fp)
                    logging.info(f'{fp} saved.')
                logging.info(f'Validation Epoch [{epoch}]/[{self.config.n_epochs}] || nmse: {loss2.item():.5f}, acc: {acc.item():.7f}, best nmse: {best_nmse:.5f}')
            
            CE2.train()
            SD.train()
            for it, (Yp4fc, Yp4cnn, Yd, X, H_label) in enumerate(train_loader): # H_train: bsx2(real and imag)x4x32
                optimizer_CE2.zero_grad()
                optimizer_SD.zero_grad()
                bs = Yp4fc.shape[0]
                Yp4fc = Yp4fc.to(device)
                Hf = FC(Yp4fc)
                Yp4fc = Yp4fc.cpu() # release gpu memory
                Hf = Hf.reshape(bs, 2, 4, 256)
                cnn_input = torch.cat([Yd.to(device), Yp4cnn.to(device), Hf], dim = 2)
                output2 = CNN(cnn_input).reshape(bs, 2, 4, 32)
                
                H_train_padding = copy.deepcopy(output2)
                output2 = output2.cpu() 
                H_train_padding = torch.cat([H_train_padding, torch.zeros(bs,2,4,256-32, requires_grad=True).to(device)],3)
                H_train_padding = H_train_padding.permute(0,2,3,1)

                H_train_padding = torch.fft(H_train_padding, 1)/20
                H_train_padding = H_train_padding.permute(0,3,1,2).contiguous() 

                X_LS = self.get_LS(Yd, H_train_padding.detach().cpu())
                X_input_train = torch.zeros([bs, 2, 2, 256], dtype = torch.float32)
                X_input_train[:,0,:,:] = torch.tensor(X_LS.real, dtype = torch.float32)
                X_input_train[:,1,:,:] = torch.tensor(X_LS.imag, dtype = torch.float32)

                input3 = torch.cat([X_input_train.cuda(), Yp4cnn.to(device), Yd.cuda(), H_train_padding.to(device)], dim = 2)
                output3 = SD(input3)

                loss1 = nn.BCELoss()(output3, X.to(device))

                output3 = output3.reshape([bs,2,256,2])
                output3 = output3.permute(0,3,1,2).contiguous()

                # input4 = torch.cat([X_input_test.cuda(), Yd_input_test.cuda(), H_test_padding], 2)
                input4 = torch.cat([output3.cuda(), Yd.cuda(), H_train_padding], 2)

                output4 = CE2(input4).reshape(bs, 2, 4, 32)

                loss2 = criterion(output4, H_label.to(device))
                loss = loss1+loss2
                # nmse1 = criterion(output2, H_label)
                loss.backward()
                optimizer_CE2.step()
                optimizer_SD.step()

                if it % self.config.print_freq == 0:
                    # print(nmse)
                    logging.info(f'Epoch: [{epoch}/{self.config.n_epochs}][{it}/{len(train_loader)}] || Loss {loss.item():.5f} || loss1 {loss1.item():.5f} || loss2 {loss2.item():.5f} ')

            if epoch >0:
                if epoch % self.config.lr_decay ==0:
                    optimizer_CE2.param_groups[0]['lr'] =  max(optimizer_CE2.param_groups[0]['lr'] * 0.5, self.config.lr_threshold)
                    optimizer_SD.param_groups[0]['lr'] = max(optimizer_SD.param_groups[0]['lr'] * 0.5, self.config.lr_threshold)
    
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

    def get_LS(self, Yd_input, Hf_input):
        Yd = np.array(Yd_input[:,0,:,:] + 1j*Yd_input[:,1,:,:])
        Hf = np.array(Hf_input[:,0,:,:] + 1j*Hf_input[:,1,:,:]) 
        Hf = np.reshape(Hf, [-1,2,2,256], order = 'F')
        X_LS = self.LSequalization(Hf, Yd)
        X_LS.real = (X_LS.real > 0)*2 - 1
        X_LS.imag = (X_LS.imag > 0)*2 - 1
        return X_LS