import math
import numpy as np

def optimizeThr(fp,tp,th):
    euc = lambda x, y : math.sqrt(x**2 + (y - 1)**2)
    veuc = np.vectorize(euc)
    dist = veuc(fp,tp)
    argmin = np.argmin(dist)

    print('>>> Recommended working point: ', th[argmin])

    return th[argmin], fp[argmin], tp[argmin]
