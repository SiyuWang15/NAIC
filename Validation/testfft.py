import numpy as np
import torch

### case 1 ####
# numpy形式fft
a = np.random.randn(10, 256) + 1j * np.random.randn(10, 256) 
af = np.fft.fft(a, 256)/20

# torch形式fft(必须在实数域上使用)
b = torch.zeros(10,256,2, dtype = torch.float32)
b[:,:,0] = torch.tensor(np.real(a))
b[:,:,1] = torch.tensor(np.imag(a))
b.requires_grad = True
bf1 = torch.fft(b, 1)/20


bf = bf1[:,:,0] + 1j*bf1[:,:,1]
print(np.max(np.abs(af - bf.detach().numpy())))



#### case 2 神经网络的时域输出转成频域输入####
c = np.random.randn(10, 4, 32) + 1j * np.random.randn(10, 4, 32)
cf = np.fft.fft(c, 256)/20

######### 以下重要 ########
d = torch.zeros(10,2,4,32, dtype = torch.float32) # d为常用的神经网络时域输出，维度为Batch*2*4*32
d[:,0,:,:] = torch.tensor(np.real(c))
d[:,1,:,:] = torch.tensor(np.imag(c))
d.requires_grad = True
d = torch.cat([d, torch.zeros(10,2,4,256-32, requires_grad=True)],3)
d = d.permute(0,2,3,1)
df1 = torch.fft(d, 1)/20
df1 = df1.permute(0,3,1,2) #df1为对应的频域形式，维度为Batch*2*4*256
print(df1.requires_grad)
######## 以上重要 ########


df = df1[:,0,:,:] + 1j*df1[:,1,:,:]
print(np.max(np.abs(cf - df.detach().numpy())))
print('Lovelive!')