# Exercise exr-csd-ppot-vacuum-f: at an arbitrary vacuum |Phi| = 1 the internal field strength F_12 is conjugate to
# -Q_em/2: its eigenvalues are (i/6)(-2, 1, 1), so V = -Tr F_12^2 = 1/6 on the whole vacuum manifold.
E = lambda i, j: matrix(QQbar, 3, 3, lambda a, b: 1 if (a, b) == (i, j) else 0)
ad = lambda A, B: A * B - B * A
tau = I / 3 * diagonal_matrix(QQbar, [1, 1, -2])
def F12(p, q):
    Pp = p * E(0, 2) + q * E(1, 2); Pm = -Pp.conjugate_transpose()
    f1 = (Pp + Pm) / 2; f2 = I * (Pp - Pm) / 2
    return ad(f1, f2) - tau
vacua = [(QQbar(3) / 5, QQbar(4) / 5 * I), ((1 + I) / 2, (1 - I) / 2), (QQbar(1), QQbar(0)), (QQbar(5) / 13 * I, QQbar(-12) / 13)]
target = sorted([-2 * I / 6, I / 6, I / 6], key=lambda z: z.imag())
ok = True
for p, q in vacua:
    F = F12(p, q); ev = sorted(F.charpoly().roots(QQbar, multiplicities=False), key=lambda z: z.imag())
    mult = sorted([m for _, m in F.charpoly().roots(QQbar)])
    print("Phi =", (p, q), ": |Phi|^2 =", p * p.conjugate() + q * q.conjugate(), ", eigenvalues of F_12:", ev, "multiplicities", mult, ", V =", -(F * F).trace())
    ok = ok and bool(p * p.conjugate() + q * q.conjugate() == 1) and ev == sorted([-2 * I / 6, I / 6], key=lambda z: z.imag()) and mult == [1, 2] and -(F * F).trace() == QQbar(1) / 6
print("[PASS] on every sampled vacuum F_12 has eigenvalues (i/6)(-2, 1, 1), the spectrum of -Q_em/2, and V = 1/6:", ok)
