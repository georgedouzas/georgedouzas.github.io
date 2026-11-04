# Exercise exr-lg-killing-su-n: the Killing form of su(n) is kappa(X, Y) = 2n tr(XY), computed from the adjoint
# matrices in a real basis of the traceless anti-Hermitian matrices, for n = 2 and n = 3.
K.<i> = QuadraticField(-1)
def su_basis(n):
    B = []
    for j in range(n - 1):
        B.append(matrix(K, n, n, {(j, j): i, (j + 1, j + 1): -i}))
    for j in range(n):
        for k in range(j + 1, n):
            B.append(matrix(K, n, n, {(j, k): 1, (k, j): -1}))
            B.append(matrix(K, n, n, {(j, k): i, (k, j): i}))
    return B
def realvec(X): return vector(QQ, sum([[e[0], e[1]] for e in X.list()], []))
for n in [2, 3]:
    B = su_basis(n)
    M = matrix(QQ, [realvec(b) for b in B]).transpose()
    def ad(X): return matrix(QQ, [M.solve_right(realvec(X * b - b * X)) for b in B]).transpose()
    ads = [ad(b) for b in B]
    kap = matrix(QQ, len(B), len(B), lambda a, b: (ads[a] * ads[b]).trace())
    tr = matrix(QQ, len(B), len(B), lambda a, b: QQ((B[a] * B[b]).trace()))
    print(f"su({n}): dimension {len(B)}; kappa = {2*n} tr(XY): {kap == 2 * n * tr}; negative definite: {(-kap).is_positive_definite()}")
T = [matrix(K, [[0, -i/2], [-i/2, 0]]), matrix(K, [[0, -1/2], [1/2, 0]]), matrix(K, [[-i/2, 0], [0, i/2]])]
print("su(2) in the basis T_a: kappa(T_a, T_b) = 4 tr(T_a T_b) =", [4 * (T[a] * T[b]).trace() for a in range(3) for b in range(3)])
