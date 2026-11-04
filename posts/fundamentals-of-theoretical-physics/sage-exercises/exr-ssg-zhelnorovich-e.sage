# Exercise exr-ssg-zhelnorovich-e: in 2nu Euclidean dimensions take Dirac matrices gamma_1..gamma_2nu (squares +1) of which
# nu are symmetric and nu antisymmetric (Pauli products). Zhelnorovich's matrix E with gamma_i^T = -E gamma_i E^{-1} is the
# product of the symmetric ones for even nu and of the antisymmetric ones for odd nu (his eqs. 1.51-1.52). We check
#   E^T = (-1)^{nu(nu+1)/2} E                                     (1.53)
#   (E gamma_{i1...ik})^T = (-1)^{[nu(nu+1) + k(k+1)]/2} E gamma_{i1...ik}   (1.55)
# for nu = 1, ..., 5 and every k, with gamma_{i1...ik} the antisymmetrised products (for distinct indices, the plain product).
from itertools import combinations
K.<i> = QuadraticField(-1)
s1 = matrix(K, [[0, 1], [1, 0]]); s2 = matrix(K, [[0, -i], [i, 0]]); s3 = matrix(K, [[1, 0], [0, -1]]); one = identity_matrix(K, 2)
def kron(ms):
    out = matrix(K, [[1]])
    for m in ms: out = out.tensor_product(m)
    return out
def euclid(nu):
    e = []
    for k in range(nu):
        e += [kron([s3] * k + [s1] + [one] * (nu - k - 1)), kron([s3] * k + [s2] + [one] * (nu - k - 1))]
    return e
allok = True
for nu in range(1, 6):
    g = euclid(nu); N = 2^nu
    sym = [x for x in g if x.transpose() == x]; asym = [x for x in g if x.transpose() == -x]
    E = prod(sym if nu % 2 == 0 else asym, identity_matrix(K, N))
    ok_def = all(x.transpose() == -E * x * E.inverse() for x in g)
    sE = 1 if E.transpose() == E else -1
    ok53 = sE == (-1)^(nu * (nu + 1) // 2)
    ok55 = True
    for k in range(0, 2 * nu + 1):
        for idx in list(combinations(range(2 * nu), k))[:6]:        # a few index sets for each k
            M = E * prod([g[j] for j in idx], identity_matrix(K, N))
            ok55 &= M.transpose() == (-1)^((nu * (nu + 1) + k * (k + 1)) // 2) * M
    print("2nu = %2d: gamma^T = -E gamma E^-1: %s; E is %s; symmetry rule (1.55) for k = 0..%d: %s"
          % (2 * nu, ok_def, "symmetric" if sE == 1 else "antisymmetric", 2 * nu, ok55))
    allok &= ok_def and ok53 and ok55
print("[PASS] E exists, E^T = (-1)^{nu(nu+1)/2} E and (E gamma_{i1..ik})^T = (-1)^{[nu(nu+1)+k(k+1)]/2} E gamma_{i1..ik} for 2nu = 2, ..., 10:", allok)
