from utils import *

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
            # mode = 0
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
def generatorXY(batch, H, Pilotnum):
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
        # SNRdb = 10
        mode = np.random.randint(0, 3)
        # mode = 0
        # mode = 2
        YY = MIMO(X, HH, SNRdb, mode, Pilotnum) / 20  ###
        XX = np.concatenate((bits0, bits1), 0)
        input_labels.append(XX)
        input_samples.append(YY)
        input_channels.append(HH)

    batch_y = np.asarray(input_samples)
    batch_x = np.asarray(input_labels)
    batch_h = np.asarray(input_channels)
    return batch_y, batch_x, batch_h
