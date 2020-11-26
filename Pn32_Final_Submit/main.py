from submit import *
import os
import numpy as np 
import logging

def ensemble():
    d = 'results'
    Pn = 32
    fps = [f for f in os.listdir(d) if f.split('.')[-1] == 'bin' and f.split('.')[0] != 'X_pre_2']
    print(fps)
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
    fp = f'./X_pre_{tag}.bin'
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

if not os.path.isdir('results'):
    os.makedirs('./results')
# 加载用于测试的接收数据 Y shape=[batch*2048]
Y = np.loadtxt('data/Y_1.csv', dtype=np.float32, delimiter=',')


print('Begin generating X_pre_1_1.bin')
submit1(Y)
print('==============================')

print('Begin generating X_pre_1_2.bin')
submit2(Y)
print('==============================')

print('Begin generating X_pre_1_3.bin')
submit3(Y)
print('==============================')

print('Begin generating X_pre_1_4.bin')
submit4(Y)
print('==============================')

print('Begin generating X_pre_1_5.bin')
submit5(Y)
ensemble()
