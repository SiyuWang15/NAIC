import torch
import numpy as np
def generate_confidence(x):
    # 输入：
    # x: batch*2*2*256
    batch = x.size(0)
    y = torch.abs(x - 0.5) * 2
    y = y.reshape(batch,4,256)
    y = torch.prod(y, 1)
    y = y.reshape(batch, 128,2)
    y = torch.prod(y, 2)
    return y

if __name__ =='__main__':
    data = np.load('X.npy')
    data = torch.tensor(data)
    conf = generate_confidence(data)
    print('Love Live')