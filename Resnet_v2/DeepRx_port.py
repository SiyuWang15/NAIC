import numpy as np
import DeepRx_model as model
import torch

#SD_path = '/home/hjiang/AI second part/Modelsave0_p'+str(Pilot_num)+'/Rhy5_f4_'

def DeepRx(SD_path, cuda_gpu=True, gpus = []):
    nn = 2
    num_nn = 8
    SD = model.ini_weights(nn, num_nn, cuda_gpu, gpus)
    SD = model.Load_Model(SD, nn, SD_path)
    return SD

# H_hat.shape； [Ns,2,2,2,256]
# Yd.shape； [Ns,2,2,256]
def Pred_DeepRX(SD, H_hat, Yd):
    Ns = Yd.shape[0]
    X_pred = []
    num_nn = 8
    for i in range(num_nn):
        idx_y = model.Idx_nn(num_nn,i,'y')
        idx_p = model.Idx_nn(num_nn,i,'p')
        k,j = divmod(i,int(num_nn/2))
        pred = SD[k](H_hat[:,:,:,:,idx_p], Yd[:,:,:,idx_p])
        X_pred.append(pred)
    X_pred = torch.reshape(torch.stack(X_pred, dim=1),[Ns,1024])
    return X_pred