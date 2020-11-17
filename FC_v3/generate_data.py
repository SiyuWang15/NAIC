from utils import *
import struct
####################使用链路和信道数据产生训练数据##########
def generator(batch,H,Pilotnum):
    while True:
        input_labels = []
        input_samples = []
        input_channels = []
        for row in range(0, batch):
            bits0 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
            bits1 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
            X=[bits0, bits1]
            temp = np.random.randint(0, len(H))
            HH = H[temp]
            SNRdb = np.random.uniform(8, 12)
            mode = np.random.randint(0, 3)
            YY = MIMO(X, HH, SNRdb, mode,Pilotnum)/20 ###
            XX = np.concatenate((bits0, bits1), 0)
            input_labels.append(XX)
            input_samples.append(YY)
            input_channels.append(HH)
        batch_y = np.asarray(input_samples)
        batch_x = np.asarray(input_labels)
        batch_h = np.asarray(input_channels)
        yield (batch_y, batch_x, batch_h)

########产生测评数据，仅供参考格式##########
def generatorXY(batch, H, Pilot_num,mode=-1):
    input_labels = []
    input_samples = []
    input_channels = []

    for row in range(0, batch):
        bits0 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
        bits1 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
        X = [bits0, bits1]
        # temp = np.random.randint(0, len(H))
        temp = row
        HH = H[temp]
        SNRdb = np.random.uniform(8, 12)
        if mode == -1:
            mode = np.random.randint(0, 3)
            # print(mode)
        YY = MIMO(X, HH, SNRdb, mode, Pilot_num) / 20  ###
        XX = np.concatenate((bits0, bits1), 0)
        input_labels.append(XX)
        input_samples.append(YY)
        input_channels.append(HH)

    batch_y = np.asarray(input_samples)
    batch_x = np.asarray(input_labels)
    batch_h = np.asarray(input_channels)
    return batch_y, batch_x, batch_h

if __name__ == '__main__':
    # channel data for training and validation
    data1 = open('/data/CuiMingyao/AI_competition/OFDMReceiver/H.bin', 'rb')
    H1 = struct.unpack('f' * 2 * 2 * 2 * 32 * 320000, data1.read(4 * 2 * 2 * 2 * 32 * 320000))
    H1 = np.reshape(H1, [320000, 2, 4, 32])
    H_tra = H1[:, 1, :, :] + 1j * H1[:, 0, :, :]  # time-domain channel for training

    data2 = open('/data/CuiMingyao/AI_competition/OFDMReceiver/H_val.bin', 'rb')
    H2 = struct.unpack('f' * 2 * 2 * 2 * 32 * 2000, data2.read(4 * 2 * 2 * 2 * 32 * 2000))
    H2 = np.reshape(H2, [2000, 2, 4, 32])
    H_val = H2[:, 1, :, :] + 1j * H2[:, 0, :, :]  # time-domain channel for training

    Pilot_num = 8
    # SNRdb = -1
    mode = -1
    Y,X,H = generatorXY(320000,H_tra,Pilot_num=Pilot_num, mode=mode)
    np.save('/data/CuiMingyao/AI_competition/OFDMReceiver/training_Y_P='+str(Pilot_num)+'_mode='+str(mode)+'.npy', Y)


    Y, X, H = generatorXY(2000, H_val, Pilot_num=Pilot_num, mode=mode)
    np.save('/data/CuiMingyao/AI_competition/OFDMReceiver/validation_Y_P='+str(Pilot_num)+'_mode='+str(mode)+'.npy', Y)