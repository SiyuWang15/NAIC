import os
import sys
import numpy as np
import logging
import submit.Submit_CE_FC_CNN_SD_HS_CE2_CNN_NHSCE_pilot8 as S1
import submit.Submit_CE_FC_CNN_SD_LS_XYH_CE2_CNN_NSDCE_pilot8 as S2
import submit.Submit_CE_FC_CNN_SD_LS_XYH_CE2_CNN_NSoftMLCE_pilot8 as S3
import submit.Submit_CE_FC_Resnet34_SD_HS_CE2_CNN_NHSCE_pilot8 as S4
import submit.Submit_CE_FC_Resnet34_SD_LS_XYH_CE2_CNN_NSoftMLCE_pilot8 as S5
import validation.Validation_CE_FC_CNN_SD_HS_CE2_CNN_NHSCE_pilot8 as V1
import validation.Validation_CE_FC_CNN_SD_LS_XYH_CE2_CNN_NSDCE_pilot8 as V2




def generate_X_pre2(prefix = './', gpu_list = '0,1'):
    print('-'*10, 'Begin Generating!','-'*10)
    # 1
    print('-'*10, 'Generate S1','-'*10)
    S1.run(prefix= prefix, gpu_list= gpu_list, N = 3)
    print('-'*10, 'Generate S1 Successfully','-'*10)
    # 2
    print('-'*10, 'Generate S2','-'*10)
    S2.run(prefix= prefix, gpu_list= gpu_list, N = 3)
    print('-'*10, 'Generate S2 Successfully','-'*10)
    # 3
    print('-'*10, 'Generate S3','-'*10)
    S3.run(prefix= prefix, gpu_list= gpu_list, N = 2)
    print('-'*10, 'Generate S3 Successfully','-'*10)
    # 4
    print('-'*10, 'Generate S4','-'*10)
    S4.run(prefix= prefix, gpu_list= gpu_list, N = 2)
    print('-'*10, 'Generate S4 Successfully','-'*10)
    # 5
    print('-'*10, 'Generate S5','-'*10)
    S5.run(prefix= prefix, gpu_list= gpu_list, N = 3)
    print('-'*10, 'Generate S5 Successfully','-'*10)


def merge():
    d = './results/'
    fps = [f for f in os.listdir(d) if f.split('.')[-1] == 'bin']
    Pn = 8
    threshold = int((len(fps) / 2))


    def set_logger():
        log = os.path.join(f'{d}/log')
        level = getattr(logging, 'INFO', None)
        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(log)
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)
        logging.info('log at: {}'.format(log))

    set_logger()
    predx = []
    for i in range(len(fps)):
        predx.append(np.fromfile(f'{d}/{fps[i]}', dtype=np.bool).astype(np.float))
    predx = np.stack(predx, axis = 0)
    logging.info(f"THRESHOLD IS {threshold}!!!!!!!!!!!!")
    x = (predx.sum(axis = 0) > threshold)

    tag = 1 if Pn == 32 else 2
    fp = f'X_pre_{tag}.bin'
    x.tofile(fp)
    logging.info(f'{fp} saved.')

    f = open(f'./{d}/sim', 'w')
    fps.append('final')
    predx = np.concatenate([predx, np.expand_dims(x, 0)], axis = 0)
    infom = []
    for i in range(-1, len(fps)):
        if i==-1:
            s = '{:<6}'.format('sim')
            for j in range(len(fps)):
                s += f'\t{j:>6}'
        else:
            s = f'{i:<6}'
            for j in range(len(fps)):
                s += f'\t{(predx[i, :] == predx[j,:]).mean():.5f}'
        infom.append(s)
    for s in infom:
        f.write(s+'\n')
    

if __name__ == '__main__':
    
    # generate_X_pre2(gpu_list= '3,7') 
    # merge()
    # V1.run(prefix= './', gpu_list= '3,7')   
    V2.run(prefix = './', gpu_list= '3,7')

