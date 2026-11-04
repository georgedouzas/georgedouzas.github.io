# Exercise exr-ssg-gamma-construction: Dirac matrices in d dimensions with {Gamma^a, Gamma^b} = 2 eta^{ab},
# eta = diag(1, -1, ..., -1), built as tensor products of Pauli matrices (the standard recursive construction). For
# even d the matrices are 2^{d/2}-dimensional; for odd d one adds the chirality matrix of d - 1. We check d = 4, 5, 10, 11:
# the eleven-dimensional matrices are 32 x 32, and Gamma^0 Gamma^1 ... Gamma^10 is proportional to the identity.
s1 = matrix(QQbar, [[0, 1], [1, 0]]); s2 = matrix(QQbar, [[0, -I], [I, 0]]); s3 = matrix(QQbar, [[1, 0], [0, -1]]); one = identity_matrix(QQbar, 2)
def kron(*ms):
    out = matrix(QQbar, [[1]])
    for m in ms: out = out.tensor_product(m)
    return out
def euclid(n):
    """2n Euclidean gamma matrices (squares +1) of size 2^n"""
    gs = []
    for k in range(n):
        pre = [s3] * k; post = [one] * (n - k - 1)
        gs += [kron(*(pre + [s1] + post)), kron(*(pre + [s2] + post))]
    return gs
def lorentz(d):
    n = d // 2; e = euclid(n)
    if d % 2 == 1:
        e = e + [prod(e) * (I^n if True else 1)]          # the product of all 2n, normalised below to square to +1
        x = e[-1]; c = (x * x)[0, 0]; e[-1] = x / sqrt(QQbar(c))
    # Gamma^0 = e_0 (squares +1), spatial Gamma^k = i e_k (square -1)
    return [e[0]] + [I * g for g in e[1:]]
def check(d):
    G = lorentz(d); N = G[0].nrows()
    eta = [1] + [-1] * (d - 1)
    ok = all(G[a] * G[b] + G[b] * G[a] == (2 * eta[a] if a == b else 0) * identity_matrix(QQbar, N) for a in range(d) for b in range(d))
    return G, N, ok
for d in (4, 5, 10, 11):
    G, N, ok = check(d)
    print("[PASS] d = %d: %d matrices of size %d = 2^floor(d/2) satisfy {Gamma^a, Gamma^b} = 2 eta^{ab}:" % (d, d, N), ok and N == 2^(d // 2) and len(G) == d)
G, N, ok = check(11)
P = prod(G)
print("[PASS] in d = 11 the product Gamma^0 ... Gamma^10 is a multiple of the identity (so no chirality matrix):", P == P[0, 0] * identity_matrix(QQbar, 32) and P[0, 0] != 0)
print("    Gamma^0 ... Gamma^10 =", P[0, 0], "* identity")
