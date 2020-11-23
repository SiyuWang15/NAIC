from utils import *
from generate_data import *
import struct

data2 = open('/data/CuiMingyao/AI_competition/OFDMReceiver/H_val.bin','rb')
H2 = struct.unpack('f'*2*2*2*32*2000,data2.read(4*2*2*2*32*2000))
H2 = np.reshape(H2,[2000,2,4,32])
H_val = H2[:,1,:,:]+1j*H2[:,0,:,:]
Pilotnum = 8
SNRdB = -1
mode = 0

# test_data = generatorXY(2000,H_val,Pilotnum,SNRdB, mode)
# Y_test, X_test, H_test = test_data

P = 64
K = 256
Pilot_file_name = 'Pilot_' + str(P)
bits = np.loadtxt(Pilot_file_name, delimiter=',')
mu = 2
pilotValue = Modulation(bits, mu)
allCarriers = np.arange(K)
pilotCarriers = np.arange(0, K, K // P)
dataCarriers = [val for val in allCarriers if not (val in pilotCarriers)]
pilotCarriers1 = pilotCarriers[0:P:2]
pilotCarriers2 = pilotCarriers[1:P:2]
pilotValue1 = pilotValue[0:P:2]
pilotValue2 = pilotValue[1:P:2]

OFDM_data1 = np.zeros(K, dtype=complex)
OFDM_data1[pilotCarriers1] = pilotValue1
OFDM_data2 = np.zeros(K, dtype=complex)
OFDM_data2[pilotCarriers2] = pilotValue2

OFDM_data1 = OFDM_data1.reshape([1,K])
OFDM_data2 = OFDM_data2.reshape([1,K])

OFDM_data = np.concatenate([OFDM_data1, OFDM_data2], 0)

Pilot_bits = np.reshape(OFDM_data, [2 , 256, 1])
Pilot_bits = np.concatenate([np.real(Pilot_bits)>0,np.imag(Pilot_bits)>0], 2)*1
Pilot_bits = np.reshape(Pilot_bits, [1, 1024])


if P == 64:
    X_1 = np.array(np.floor(Pilot_bits + 0.5), dtype=np.bool)
    X_1.tofile('./X_Pilot_64.bin')

if P == 16:
    X_1 = np.array(np.floor(Pilot_bits + 0.5), dtype=np.bool)
    X_1.tofile('./X_Pilot_16.bin')

print('Love Live')