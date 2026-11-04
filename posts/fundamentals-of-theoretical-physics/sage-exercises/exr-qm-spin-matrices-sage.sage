# Exercise exr-qm-spin-matrices-sage: spin-s matrices for s = 1/2, 1, 3/2, 2, their algebra, Casimir and spectrum of S_1.
hbar = var("hbar", domain="positive")
def spin(s):
    ms = [s - k for k in range(int(2 * s) + 1)]; d = len(ms)
    Sz = hbar * diagonal_matrix(SR, ms)
    Sp = matrix(SR, d, d, lambda i, j: hbar * sqrt(s * (s + 1) - ms[j] * (ms[j] + 1)) if i == j - 1 else 0)
    Sm = Sp.conjugate_transpose()
    return (Sp + Sm) / 2, (Sp - Sm) / (2 * I), Sz
for s in [1 / 2, 1, 3 / 2, 2]:
    S1, S2, S3 = spin(s)
    z = lambda M: M.simplify_full().is_zero()
    ok = z(S1 * S2 - S2 * S1 - I * hbar * S3) and z(S2 * S3 - S3 * S2 - I * hbar * S1) and z(S3 * S1 - S1 * S3 - I * hbar * S2)
    cas = (S1 ** 2 + S2 ** 2 + S3 ** 2).simplify_full()
    print("s =", s, " algebra:", ok, " S^2 = hbar^2 s(s+1):", cas == hbar ** 2 * s * (s + 1) * identity_matrix(S3.nrows()),
          " eigenvalues of S_1 / hbar:", sorted((S1 / hbar).eigenvalues()))
print("S_+ for s = 3/2:"); print((spin(3 / 2)[0] + I * spin(3 / 2)[1]).simplify_full() / hbar)
