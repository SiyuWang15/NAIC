from utils import *
import struct

from scipy.io import loadmat
from scipy.io import savemat





####################使用链路和信道数据产生训练数�?#########
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
        YY = MIMO(X, HH, SNRdb, mode,Pilotnum) ###
        YY = YY/20
        XX = np.concatenate((bits0, bits1), 0)
        input_labels.append(XX)
        input_samples.append(YY)
        input_channels.append(HH)
    batch_y = np.asarray(input_samples)
    batch_x = np.asarray(input_labels)
    batch_h = np.asarray(input_channels)
    return batch_y, batch_x, batch_h


###################使用LS估计有导频的频域信道##########
def LS_Estimation(Y,Pilotnum):
    K = 256
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

    allCarriers = np.arange(K)
    pilotCarriers = np.arange(0, K, K // P)
    pilotCarriers1 = pilotCarriers[0:P:2]
    pilotCarriers2 = pilotCarriers[1:P:2]

    ## Load PilotValue
    PilotValue_file_name = 'PilotValue_' + str(Pilotnum) + '.mat'
    D = loadmat(PilotValue_file_name)
    PilotValue = D['P']  ## 1*Pilotnum*2
    PilotValue1 = PilotValue[:,0:P:2] ## Pilot value for the first transmitting antenna
    PilotValue2 = PilotValue[:,1:P:2] ## Pilot value for the second transmitting antenna

    ## LS estimation
    Hf = np.zeros(shape=(Ns, 2, 2, 256),dtype=np.complex64)

    Hf[:, 0, 0, pilotCarriers1] = Yp1[:, pilotCarriers1]/PilotValue1
    Hf[:, 0, 1, pilotCarriers2] = Yp1[:, pilotCarriers2]/PilotValue2
    Hf[:, 1, 0, pilotCarriers1] = Yp2[:, pilotCarriers1]/PilotValue1
    Hf[:, 1, 1, pilotCarriers2] = Yp2[:, pilotCarriers2]/PilotValue2

    return Hf


def Interpolation_f(Hf, Pilotnum):
    Ns = Hf.shape[0]
    Hf_inter = np.zeros((Ns, 2, 2, 256), dtype=np.complex64)
    if Pilotnum == 32:
        i ,j = np.meshgrid(np.arange(256), np.arange(256))
        omega = np.exp(-2*np.pi*1j/256)
        DFT_Matrix = np.power(omega, i*j)

        inP1 = int(256 / Pilotnum)
        inP2 = int(128 / Pilotnum)
        idx1 = np.arange(0,256,inP1)
        idx2 = np.arange(inP2,256,inP1)
        DFT_1 = DFT_Matrix[idx1,0:32]
        DFT_1v = np.linalg.inv(DFT_1)
        DFT_2 = DFT_Matrix[idx2,0:32]
        DFT_2v = np.linalg.inv(DFT_2)
        hf_00 = Hf[:, 0, 0, idx1].T
        hf_01 = Hf[:, 0, 1, idx2].T
        hf_10 = Hf[:, 1, 0, idx1].T
        hf_11 = Hf[:, 1, 1, idx2].T
        ht_00 = np.dot(DFT_1v, hf_00).T
        ht_01 = np.dot(DFT_2v, hf_01).T
        ht_10 = np.dot(DFT_1v, hf_10).T
        ht_11 = np.dot(DFT_2v, hf_11).T
        Hf_inter[:, 0, 0, :] = np.fft.fft(ht_00,256)
        Hf_inter[:, 0, 1, :] = np.fft.fft(ht_01,256)
        Hf_inter[:, 1, 0, :] = np.fft.fft(ht_10,256)
        Hf_inter[:, 1, 1, :] = np.fft.fft(ht_11,256)
    elif Pilotnum == 8:
        i ,j = np.meshgrid(np.arange(256), np.arange(256))
        omega = np.exp(-2*np.pi*1j/256)
        DFT_Matrix = np.power(omega, i*j)

        inP1 = int(256 / Pilotnum)
        inP2 = int(128 / Pilotnum)
        idx1 = np.arange(0,256,inP1)
        idx2 = np.arange(inP2,256,inP1)
        idx1_inter = np.arange(0, 256, int(inP1/4))
        idx2_inter = np.arange(int(inP2/4), 256, int(inP1/4))
        DFT_1 = DFT_Matrix[idx1_inter,0:32]
        DFT_1v = np.linalg.inv(DFT_1)
        DFT_2 = DFT_Matrix[idx2_inter,0:32]
        DFT_2v = np.linalg.inv(DFT_2)
        hf_00 = Hf[:, 0, 0, idx1]
        hf_01 = Hf[:, 0, 1, idx2]
        hf_10 = Hf[:, 1, 0, idx1]
        hf_11 = Hf[:, 1, 1, idx2]
        hf_00_inter = hf_01_inter = hf_10_inter = hf_11_inter = np.zeros((Ns, 32),dtype=np.complex64)
        for sample in range(0, Ns):
            hf_00_inter[sample, :] = np.interp(idx1_inter,  idx1, hf_00[sample, :])
            hf_01_inter[sample, :] = np.interp(idx2_inter,  idx2, hf_01[sample, :])
            hf_10_inter[sample, :] = np.interp(idx1_inter,  idx1, hf_10[sample, :])
            hf_11_inter[sample, :] = np.interp(idx2_inter,  idx2, hf_11[sample, :])
        ht_00_inter = np.dot(DFT_1v, hf_00_inter.T).T
        ht_01_inter = np.dot(DFT_2v, hf_01_inter.T).T
        ht_10_inter = np.dot(DFT_1v, hf_10_inter.T).T
        ht_11_inter = np.dot(DFT_2v, hf_11_inter.T).T
        Hf_inter[:, 0, 0, :] = np.fft.fft(ht_00_inter,256)
        Hf_inter[:, 0, 1, :] = np.fft.fft(ht_01_inter,256)
        Hf_inter[:, 1, 0, :] = np.fft.fft(ht_10_inter,256)
        Hf_inter[:, 1, 1, :] = np.fft.fft(ht_11_inter,256)
    else:
        print('error!')
    return Hf_inter

###################使用LS估计有导频的频域信道##########
def LS_Estimation_Partial(Y,Pilotnum):

    K = 256
    P = Pilotnum*2
    Ns = Y.shape[0]

    Y = np.reshape(Y, [-1, 2, 2, 2, 256], order='F')
    Y_complex = Y[:,0,:,:,:]+1j * Y[:,1,:,:,:]
    Yp = Y_complex[:,0,:,:] # Received pilot signal
    Yp1 = Yp[:,0,:] # Received pilot signal for the first receiving antenna
    Yp2 = Yp[:,1,:] # Received pilot signal for the second receiving antenna


    ## Load PilotValue
    PilotValue_file_name = 'PilotValue_' + str(Pilotnum) + '.mat'
    D = loadmat(PilotValue_file_name)
    PilotValue = D['P']  ## 1*Pilotnum*2
    PilotValue1 = PilotValue[:,0:P:2] ## Pilot value for the first transmitting antenna
    PilotValue2 = PilotValue[:,1:P:2] ## Pilot value for the second transmitting antenna

    ## LS estimation
    Hf = np.zeros(shape=(Ns, 2, 2, Pilotnum),dtype=np.complex64)

    pilotCarriers = np.arange(0, K, K // P)
    pilotCarriers1 = pilotCarriers[0:P:2]
    pilotCarriers2 = pilotCarriers[1:P:2]

    Hf[:, 0, 0, :] = Yp1[:, pilotCarriers1]/PilotValue1
    Hf[:, 0, 1, :] = Yp1[:, pilotCarriers2]/PilotValue2
    Hf[:, 1, 0, :] = Yp2[:, pilotCarriers1]/PilotValue1
    Hf[:, 1, 1, :] = Yp2[:, pilotCarriers2]/PilotValue2

    return Hf


def MMSE_Estimation(Y,Pilotnum):

    K = 256 # the number of subcarriers
    P = Pilotnum*2
    Ns = Y.shape[0]

    Y = np.reshape(Y, [-1, 2, 2, 2, 256], order='F')
    Y_complex = Y[:,0,:,:,:]+1j * Y[:,1,:,:,:]
    Yp = Y_complex[:,0,:,:] # Received pilot signal
    Yp1 = Yp[:,0,:] # Received pilot signal for the first receiving antenna
    Yp2 = Yp[:,1,:] # Received pilot signal for the second receiving antenna

    allCarriers = np.arange(K)
    pilotCarriers = np.arange(0, K, K // P)
    pilotCarriers1 = pilotCarriers[0:P:2]
    pilotCarriers2 = pilotCarriers[1:P:2]
    noiseCarries = [val for val in allCarriers if not (val in pilotCarriers)]

    # estimate the sigma2 for noise
    noise1 = Yp1[:,noiseCarries]
    noise2 = Yp2[:,noiseCarries]

    noise = np.concatenate((noise1, noise2),1)
    sigma2n = np.mean(abs(noise)**2,1)
    # sigma2n = np.mean(sigma2n)

    ## Load PilotValue
    PilotValue_file_name = 'PilotValue_' + str(Pilotnum) + '.mat'
    D = loadmat(PilotValue_file_name)
    PilotValue = D['P']  ## 1*Pilotnum*2
    PilotValue1 = PilotValue[:,0:P:2] ## Pilot value for the first transmitting antenna
    PilotValue2 = PilotValue[:,1:P:2] ## Pilot value for the second transmitting antenna

    Hf = np.zeros(shape=(Ns, 2, 2, 256),dtype=np.complex64)

    ## LS estimation
    Hf[:, 0, 0, pilotCarriers1] = Yp1[:, pilotCarriers1]/PilotValue1
    Hf[:, 0, 1, pilotCarriers2] = Yp1[:, pilotCarriers2]/PilotValue2
    Hf[:, 1, 0, pilotCarriers1] = Yp2[:, pilotCarriers1]/PilotValue1
    Hf[:, 1, 1, pilotCarriers2] = Yp2[:, pilotCarriers2]/PilotValue2

    #### load Rh for for frequency-domian channel ####
    D = loadmat('Rh_'+str(Pilotnum)+'.mat')
    Rh00 = D['Rh00']
    Rh01 = D['Rh01']
    Rh10 = D['Rh10']
    Rh11 = D['Rh11']

    ## MMSE estimation
    for i in range(Ns):
        Hf[i, 0, 0, pilotCarriers1] = Rh00 * 1 / (Rh00 + sigma2n[i]) * Hf[i, 0, 0, pilotCarriers1]
        Hf[i, 0, 1, pilotCarriers2] = Rh01 * 1 / (Rh01 + sigma2n[i]) * Hf[i, 0, 1, pilotCarriers2]
        Hf[i, 1, 0, pilotCarriers1] = Rh10 * 1 / (Rh10 + sigma2n[i]) * Hf[i, 1, 0, pilotCarriers1]
        Hf[i, 1, 1, pilotCarriers2] = Rh11 * 1 / (Rh11 + sigma2n[i]) * Hf[i, 1, 1, pilotCarriers2]


    return Hf


def Interpolation_t_weighting(Hf_partial, Pilotnum):
    ## Hf_partial: Ns*2*2*Pilotnum  有导频位置处基于LS的信道估计
    Ns = Hf_partial.shape[0]
    # print(np.sum(abs(Hf_partial) ** 2))
    Ht_partial = np.fft.ifft(Hf_partial)
    # print(np.sum(abs(Ht_partial) ** 2))
    Hf = np.fft.fft(Ht_partial,256)

    return Hf



if __name__ == '__main__':

    Ns = 100
    mode = 2
    SNRdb = 8
    Pilotnum = 8
    K = 256
    P = Pilotnum*2
    allCarriers = np.arange(K)
    pilotCarriers = np.arange(0, K, K // P)
    pilotCarriers1 = pilotCarriers[0:P:2]
    pilotCarriers2 = pilotCarriers[1:P:2]

    # data1 = open('H.bin', 'rb')
    # H1 = struct.unpack('f' * 2 * 2 * 2 * 32 * 320000, data1.read(4 * 2 * 2 * 2 * 32 * 320000))
    # H1 = np.reshape(H1, [320000, 2, 4, 32])
    # H = H1[:, 1, :, :] + 1j * H1[:, 0, :, :]

    data2 = open('H_val.bin', 'rb')
    H2 = struct.unpack('f' * 2 * 2 * 2 * 32 * 2000, data2.read(4 * 2 * 2 * 2 * 32 * 2000))
    H2 = np.reshape(H2, [2000, 2, 4, 32])
    H_val = H2[:, 1, :, :] + 1j * H2[:, 0, :, :]


    ### estimate Rh for frequency-domian channel ####
    # Ns_full = 320000
    # #
    # Hf_full = np.fft.fft(H, 256)/20
    # Hf_full = np.reshape(Hf_full,[Ns_full,2,2,256], order='F')
    # Rh00 = np.mean(Hf_full[:,0,0,pilotCarriers1]*np.conj(Hf_full[:,0,0,pilotCarriers1]),0)
    # Rh01 = np.mean(Hf_full[:,0,1,pilotCarriers2]*np.conj(Hf_full[:,0,1,pilotCarriers2]),0)
    # Rh10 = np.mean(Hf_full[:,1,0,pilotCarriers1]*np.conj(Hf_full[:,1,0,pilotCarriers1]),0)
    # Rh11 = np.mean(Hf_full[:,1,1,pilotCarriers2]*np.conj(Hf_full[:,1,1,pilotCarriers2]),0)
    # D = dict(Rh00=Rh00,Rh01=Rh01,Rh10=Rh10,Rh11=Rh11)
    # savemat('Rh_8.mat',D)

    #### load Rh for for frequency-domian channel ####
    # D = loadmat('Rh_32.mat')
    # Rh00 = D['Rh00']
    # Rh01 = D['Rh01']
    # Rh10 = D['Rh10']
    # Rh11 = D['Rh11']

    y,x,h = generator(Ns,H_val)

    Hf = np.fft.fft(h, 256)/20
    Hf = np.reshape(Hf,[Ns,2,2,256], order='F')


    # Hfper = np.zeros(shape=(Ns, 2, 2, 256), dtype=np.complex64)
    # Hfper[:, 0, 0, pilotCarriers1] = Hf[:, 0, 0, pilotCarriers1]
    # Hfper[:, 0, 1, pilotCarriers2] = Hf[:, 0, 1, pilotCarriers2]
    # Hfper[:, 1, 0, pilotCarriers1] = Hf[:, 1, 0, pilotCarriers1]
    # Hfper[:, 1, 1, pilotCarriers2] = Hf[:, 1, 1, pilotCarriers2]

    Hfper = np.zeros(shape=(Ns, 2, 2, Pilotnum), dtype=np.complex64)
    Hfper[:, :, 0, :] = Hf[:, :, 0, pilotCarriers1]
    Hfper[:, :, 1, :] = Hf[:, :, 1, pilotCarriers2]

    Hfhat_partial = LS_Estimation_Partial(y,Pilotnum)
    # Hfhat = Interpolation_t_weighting(Hfhat_partial, Pilotnum)

    # print(np.sum(abs(Hfhat)**2))
    # print(np.sum(abs(Hf) ** 2))

    # Hfhat1 = LS_Estimation(y, Pilotnum)
    # nmse = np.sum(abs(Hfhat-Hf)**2)/np.sum(abs(Hf)**2)
    # print(nmse)

    nmse = np.sum(abs(Hfhat_partial - Hfper) ** 2) / np.sum(abs(Hfper) ** 2)
    print(nmse)
    # nmse2 = np.sum(abs(Hfhat2-Hfper)**2)/np.sum(abs(Hfper)**2)
    # print(nmse2)
