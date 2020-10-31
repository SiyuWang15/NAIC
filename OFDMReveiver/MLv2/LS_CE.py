from utils import *
import struct

from scipy.io import loadmat
from scipy.io import savemat



###########################以下仅为信道数据载入和链路使用范例############

data1=open('/data/CuiMingyao/AI_competition/OFDMReceiver/H.bin','rb')
H1=struct.unpack('f'*2*2*2*32*320000,data1.read(4*2*2*2*32*320000))
H1=np.reshape(H1,[320000,2,4,32])
H=H1[:,1,:,:]+1j*H1[:,0,:,:]

Htest=H[300000:,:,:]
H=H[:300000,:,:]

####################使用链路和信道数据产生训练数据##########
def generator(batch,H):
    input_labels = []
    input_samples = []
    input_channels=[]
    for row in range(0, batch):
        bits0 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
        bits1 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
        X=[bits0, bits1]
        temp = np.random.randint(0, len(H))
        HH = H[temp]
        print(HH.shape)
        YY = MIMO(X, HH, SNRdb, mode,Pilotnum) ###
        YY = YY/20
        XX = np.concatenate((bits0, bits1), 0)
        input_labels.append(XX)
        input_samples.append(YY)
        input_channels.append(HH)
    batch_y = np.asarray(input_samples)
    batch_x = np.asarray(input_labels)
    batch_h = np.asarray(input_channels)
    print(batch_h.shape)
    return batch_y, batch_x, batch_h

# y,x,h = generator(1,H)
# print(y.shape)
# print(y.shape[0])
# print()

# D=dict(P=P)
# savemat('PilotValue_' + str(Pilotnum) + '.mat', D)

# PilotValue_file_name = 'PilotValue_' + str(Pilotnum)+ '.mat'
# D = loadmat(PilotValue_file_name)
# PilotValue = D['P']
# print(PilotValue.shape)

###################使用LS估计有导频的频域信道##########
def LS_Estimation(Y,Pilotnum):

    P = Pilotnum*2
    Ns = Y.shape[0]

    Y = np.reshape(Y, [-1, 2, 2, 2, 256], order='F')
    Y_complex = Y[:,0,:,:,:]+1j * Y[:,1,:,:,:]
    # print(Y_complex)
    Yp = Y_complex[:,0,:,:] # Received pilot signal
    # Yp = np.reshape(Yp, [Ns, 2, 256])
    # print(Yp)
    Yp1 = Yp[:,0,:] # Received pilot signal for the first receiving antenna
    Yp2 = Yp[:,1,:] # Received pilot signal for the second receiving antenna
    # print(Yp1)
    ## Load PilotValue
    PilotValue_file_name = '/PilotValue_' + str(Pilotnum) + '.mat'
    D = loadmat(PilotValue_file_name)
    PilotValue = D['P']  ## 1*Pilotnum*2
    PilotValue1 = PilotValue[:,0:P:2] ## Pilot value for the first transmitting antenna
    PilotValue2 = PilotValue[:,1:P:2] ## Pilot value for the second transmitting antenna

    ## LS estimation
    Hf = np.zeros(shape=(Ns, 2, 2, 256),dtype=np.complex64)

    inP1 = int(256 / Pilotnum)
    inP2 = int(128 / Pilotnum)

    Hf[:, 0, 0, 0:256:inP1] = Yp1[:, 0:256:inP1]/PilotValue1
    Hf[:, 0, 1, inP2:256:inP1] = Yp1[:, inP2:256:inP1]/PilotValue2
    Hf[:, 1, 0, 0:256:inP1] = Yp2[:, 0:256:inP1]/PilotValue1
    Hf[:, 1, 1, inP2:256:inP1] = Yp2[:, inP2:256:inP1]/PilotValue2

    Hf1 = np.zeros(shape=[Ns, 2, 2, 2, 256], dtype=np.float32)
    Hf_real = Hf.real
    Hf_imag = Hf.imag
    Hf1[:, 0, :, :, :] = Hf_real
    Hf1[:, 1, :, :, :] = Hf_imag

    return Hf1

Ns=2
mode = 0
SNRdb = 12
Pilotnum = 32

y,x,h = generator(Ns,H)
D=dict(y=y,x=x,h=h)
# savemat('testdata_' + str(SNRdb) + 'dB.mat', D)

Hf = np.fft.fft(h, 256)/20
Hf = np.reshape(Hf,[Ns,2,2,256], order='F')
Hfhat = LS_Estimation(y,Pilotnum)

Hfhat=Hfhat[:,0,:,:,:]+1j*Hfhat[:,1,:,:,:]

Hfper = np.zeros(shape=(Ns, 2, 2, 256), dtype=np.complex64)
inP1 = int(256 / Pilotnum)
inP2 = int(128 / Pilotnum)
Hfper[:, 0, 0, 0:256:inP1] =  Hf[:, 0, 0, 0:256:inP1]
Hfper[:, 0, 1, inP2:256:inP1] = Hf[:, 0, 1, inP2:256:inP1]
Hfper[:, 1, 0, 0:256:inP1] = Hf[:, 1, 0, 0:256:inP1]
Hfper[:, 1, 1, inP2:256:inP1] = Hf[:, 1, 1, inP2:256:inP1]

#Hfreal = np.zeros(shape=(Ns, 2, 2, 256), dtype=np.complex64)
Hfreal=Hfper



# print(np.sum(abs(Hfreal)**2))

nmse = np.sum(abs(Hfhat-Hfreal)**2)/np.sum(abs(Hfreal)**2)
print(nmse)
