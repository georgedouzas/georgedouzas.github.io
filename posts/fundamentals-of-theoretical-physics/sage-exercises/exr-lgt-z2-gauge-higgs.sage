# Exercise exr-lgt-z2-gauge-higgs: the Z2 gauge-Higgs model on the 2 x 2 torus, enumerated exactly.
import numpy as np
from itertools import product
L = 2
def averages(bG, bM):
    Z = 0; acc = np.zeros(4)
    for S in product([1, -1], repeat=4):
        S = np.array(S).reshape(L, L)
        for s in product([1, -1], repeat=8):
            s = np.array(s).reshape(2, L, L)
            P = s[0] * np.roll(s[1], -1, 0) * np.roll(s[0], -1, 1) * s[1]
            G = np.stack([S * s[mu] * np.roll(S, -1, mu) for mu in (0, 1)])
            w = np.exp(bG * P.sum() + bM * G.sum())
            Z += w; acc += w * np.array([P.mean(), G.mean(), S[0, 0], s[0, 0, 0]])
    return acc / Z
for bG, bM in ((0, 0.4), (0, 1.0), (0.5, 0.4)):
    P, G, S0, s0 = averages(bG, bM)
    print('beta_G = %.1f, beta_M = %.1f: <S s S> = %.5f, <sigma_P> = %.5f, <S_x> = %.1e, <sigma_l> = %.1e' % (bG, bM, G, P, abs(S0), abs(s0)))
    if bG == 0:
        print('       tanh(beta_M) = %.5f, tanh(beta_M)^4 = %.5f' % (np.tanh(bM), np.tanh(bM)**4))
