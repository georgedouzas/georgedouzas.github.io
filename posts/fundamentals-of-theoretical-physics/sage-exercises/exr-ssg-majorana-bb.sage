# Exercise exr-ssg-majorana-bb: in d dimensions with signature (+, -, ..., -) and the Pauli-product Dirac matrices of
# exr-ssg-gamma-construction, every Gamma^mu is either real or imaginary. A matrix B with (Gamma^mu)^* = s B Gamma^mu B^{-1}
# (s = +1 or -1, the same for all mu) is then a product of the real or of the imaginary ones. A Majorana condition
# psi^* = B psi is consistent only if B^* B = +1 (apply it twice). We list, for d = 2, ..., 11, the signs s for which B
# exists and the value of B^* B, which decides when Majorana spinors exist (Zhelnorovich, Sec. 1.6.2-1.6.3).
K.<i> = QuadraticField(-1)
s1 = matrix(K, [[0, 1], [1, 0]]); s2 = matrix(K, [[0, -i], [i, 0]]); s3 = matrix(K, [[1, 0], [0, -1]]); one = identity_matrix(K, 2)
def kron(ms):
    out = matrix(K, [[1]])
    for m in ms: out = out.tensor_product(m)
    return out
def gammas(d):
    n = d // 2
    e = []
    for k in range(n):
        e += [kron([s3] * k + [s1] + [one] * (n - k - 1)), kron([s3] * k + [s2] + [one] * (n - k - 1))]
    if d % 2:
        x = prod(e); c = (x * x)[0, 0]
        e.append(x if c == 1 else i * x)                 # chirality matrix normalised to square to +1
    return [e[0]] + [i * g for g in e[1:]]            # Gamma^0 squares to +1, Gamma^k to -1
conj = lambda M: M.apply_map(lambda z: z.conjugate())
table = {}
for d in range(2, 12):
    G = gammas(d); N = G[0].nrows()
    real = [g for g in G if conj(g) == g]; imag = [g for g in G if conj(g) == -g]
    assert len(real) + len(imag) == d
    res = []
    for B in (prod(real, identity_matrix(K, N)), prod(imag, identity_matrix(K, N))):
        for s in (1, -1):
            if all(conj(g) * B == s * B * g for g in G):
                BB = conj(B) * B
                res.append((s, BB[0, 0] if BB == BB[0, 0] * identity_matrix(K, N) else None))
    table[d] = sorted(set(res))
    print("d = %2d (d mod 8 = %d): (sign s, B^* B) = %s" % (d, d % 8, table[d]))
maj = [d for d in table if any(bb == 1 for s, bb in table[d])]
print("dimensions with B^* B = +1 for some B, so that a Majorana condition exists:", maj)
print("[PASS] Majorana spinors exist exactly for d = 2, 3, 4, 8, 9, 10, 11 among d = 2..11 (d mod 8 in {0, 1, 2, 3, 4}):", maj == [2, 3, 4, 8, 9, 10, 11])
print("[PASS] in d = 4, 10, 11 a B with B^* B = +1 exists (four-dimensional Majorana, ten-dimensional Majorana-Weyl, eleven-dimensional Majorana):",
      all(d in maj for d in (4, 10, 11)))
