# Exercise exr-htv-torus-trace: the heat trace of the torus with radii (R1, R2) = (1, 1) and (2, 1/2), its factorisation
# into theta functions, and the leading term pi R1 R2 / t.
RF = RealField(80)
theta = lambda t: sum(exp(-RF(t) * n ** 2) for n in range(-60, 61))
for R1, R2 in [(1, 1), (2, QQ(1) / 2)]:
    for t in [QQ(1) / 10, QQ(1) / 2, 2]:
        Z = sum(exp(-RF(t) * (m ** 2 / RF(R1) ** 2 + n ** 2 / RF(R2) ** 2)) for m in range(-60, 61) for n in range(-60, 61))
        print("(R1, R2) = (%s, %s), t = %s: Z = %.8f  theta theta = %.8f  pi R1 R2/t = %.8f" % (R1, R2, t, Z, theta(t / R1 ** 2) * theta(t / R2 ** 2), RF(pi) * R1 * R2 / t))
