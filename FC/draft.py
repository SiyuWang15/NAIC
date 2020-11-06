
import numpy as np
import time 





def MakeCodebook():
    G = 4
    codebook = np.zeros((G, 2**G))
    for idx in range(2**G):
        n = idx
        for i in range(G):
            r = n % 2
            codebook[G -1- i, idx] = r
            n = n//2

    return codebook

if __name__ == '__main__':

    code = MakeCodebook()

    print('Lovelive')