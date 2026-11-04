# Exercise exr-sr-boost-translation: the group commutator of the boost B(0.6 x) and the time translation by 1, with 5x5
# matrices of the Poincare group.
def poin(L, a):
    M = identity_matrix(QQ, 5); M[0:4, 0:4] = L; M[0:4, 4] = matrix(QQ, 4, 1, a); return M
B = matrix(QQ, [[5/4, 3/4, 0, 0], [3/4, 5/4, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
X, Ta = poin(B, [0, 0, 0, 0]), poin(identity_matrix(QQ, 4), [1, 0, 0, 0])
C = X*Ta*X.inverse()*Ta.inverse()
print(C)
