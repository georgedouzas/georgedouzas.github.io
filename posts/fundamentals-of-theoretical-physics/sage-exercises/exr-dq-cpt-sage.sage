# Exercise exr-dq-cpt-sage: the six orders of C psi = i gamma^2 psi^*, P psi = gamma^0 psi, T psi = i gamma^1 gamma^3 psi^* compose to
# linear maps whose matrices are multiples of gamma_5.
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
Z = zero_matrix(SR, 2); one = identity_matrix(SR, 2)
blk = lambda a, b, c, d: block_matrix([[a, b], [c, d]], subdivide=False)
g = [blk(Z, one, one, Z)] + [blk(Z, sk, -sk, Z) for sk in s]   # Weyl representation
g5 = I * g[0] * g[1] * g[2] * g[3]; I4 = identity_matrix(SR, 4)
from itertools import permutations
ops = {"C": (I * g[2], True), "P": (g[0], False), "T": (I * g[1] * g[3], True)}
def compose(order):   # apply order[0] first; an antilinear op M conj acting after a matrix X gives M X^*
    X = I4
    for name in order:
        M, anti = ops[name]
        X = M * (X.conjugate() if anti else X)
    return X
for order in permutations("TPC"):
    X = compose(order)
    c = X[0, 0] / g5[0, 0]
    print("apply", " then ".join(order), ": matrix = (%s) gamma_5:" % c, X == c * g5)
