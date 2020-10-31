import numpy as np

def MakeCodebook(G = 4):

    assert type(G) == int

    codebook = np.zeros((G, 2**G))
    for idx in range(2**G):
        n = idx
        for i in range(G):
            r = n % 2
            codebook[G -1- i, idx] = r
            n = n//2

    return codebook


def MLReceiver(Y, H, Codebook):
# Y.shape = (Batch * 2 * 2 * 256) frequency domain IQ \times Rx
# H.shape = (Batch * 2 * 2 * 256) frequency domain Tx \times Rx
# G the number subcarriers in each group \times 4(2 \times 2bit)

    G, P = Codebook.shape
    assert P == 2**G

    B = G//4
    assert B*4 == G

    T = 256//B
    assert T*B == 256

    Batch = H.shape(0)
    Y = np.reshape(Y , (Batch, 2, 2,  B, T))
    H = np.reshape(H , (Batch, 2, 2,  B, T))
    X_ML = np.zeros((Batch, 2, 2, B, T))


    for num in range(Batch):
        print('Completed batches [%d]/[%d]'%(num ,Batch))
        for idx in range(T):
            y = Y[num, :, :, :, idx]
            h = H[num, :, :, :, idx]
            y = y[0, :, :] + 1j*y[1,:,:]
            error = np.zeros((P, 1)) 

            for item in range(P):

                x = np.reshape(Codebook[:, item], (2,2,B))
                x = 0.7071 * ( 2 * x[0,...] - 1 ) + 0.7071j * ( 2 * x[1,...] - 1 )
                
                for b in range(B):
                    error[item] = error[item] + np.norm( y[:,b] - np.dot(h[:,:,b], x) )**2/2/B
            
            ML_idx = np.argmax(error)
            '''
            x_ML = np.reshape(Codebook[:, ML_idx], (2,2,B))
            x_ML = 0.7071 * ( 2 * x_ML[0,...] - 1 ) + 0.7071j * ( 2 * x_ML[1,...] - 1 )
            X_ML[num,:,:,idx] = x_ML
            '''
            X_ML[num,:,:,:,idx] = np.reshape(Codebook[:, ML_idx], (2,2,B))

    return np.reshape(X_ML, (Batch, 1024))


if __name__ == '__main__':
    
    G = 4*4

    codebook = MakeCodebook(G)

    print('Love Live!')
