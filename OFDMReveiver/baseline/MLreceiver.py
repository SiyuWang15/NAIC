import numpy as np
import time


def MakeCodebook(G=4):
    assert type(G) == int

    codebook = np.zeros((G, 2 ** G))
    for idx in range(2 ** G):
        n = idx
        for i in range(G):
            r = n % 2
            codebook[G - 1 - i, idx] = r
            n = n // 2

    return codebook


def MLReceiver(Y, H, Codebook=MakeCodebook(4)):
    # 输入：
    # Y : frequency domain 复数 (batch * 2 * 256) [样本数 * 天线数 * 子载波数]
    # H : frequency domain 复数 (batch * 2 * 2 * 256) [样本数 * 接收天线数 * 发射天线数 * 子载波数]
    # 输出：
    # X_ML : frequency domain 复数 (batch * 2 * 256) [样本数 * 发射天线数 * 子载波数]
    # X_bits    : frequency domain 比特 (batch * 1024)

    G, P = Codebook.shape
    assert P == 2 ** G

    B = G // 4
    assert B * 4 == G

    T = 256 // B
    assert T * B == 256

    batch = H.shape[0]

    # B为分组的个数，只能为2的整倍数
    Y = np.reshape(Y, (batch, 2, B, T))
    H = np.reshape(H, (batch, 2, 2, B, T))
    X_ML = np.zeros((batch, 2, B, T), dtype=complex)

    for num in range(batch):
        print('Completed batches [%d]/[%d]' % (num, batch), time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        for idx in range(T):
            y = Y[num, :, :, idx]
            h = H[num, :, :, :, idx]
            error = np.zeros((P, 1))

            for item in range(P):

                x = np.reshape(Codebook[:, item], (2, 2, B))
                x = 0.7071 * (2 * x[:, 0, :] - 1) + 0.7071j * (2 * x[:, 1, :] - 1)
                for b in range(B):
                    error[item] = error[item] + np.linalg.norm(y[:, b:b + 1] - np.dot(h[:, :, b], x[:, b:b + 1])) ** 2

            ML_idx = np.argmin(error)

            x_ML = np.reshape(Codebook[:, ML_idx], (2, 2, B))
            x_ML = 0.7071 * (2 * x_ML[:, 0, :] - 1) + 0.7071j * (2 * x_ML[:, 1, :] - 1)
            X_ML[num, :, :, idx] = x_ML
    X_ML = np.reshape(X_ML, [batch, 2, 256])


    return X_ML


if __name__ == '__main__':
    G = 4 * 4

    codebook = MakeCodebook(G)

    print('Love Live!')
