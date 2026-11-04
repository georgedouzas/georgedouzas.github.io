# Exercise exr-kkr-charge-covariant: the five-dimensional kinetic term of a complex scalar on the ansatz contains the
# covariant derivative. With the inverse ansatz (g^{mu nu} free, phi), Phi = f(x) e^{i n y/R}:
# G^{AB} d_A Phi d_B Phi^* = g^{mu nu} (d_mu - i n A_mu/R) f (d_nu + i n A_nu/R) f^* - phi^{-2} (n/R)^2 |f|^2.
# The derivatives d_mu f, d_mu f^* and f, f^* are independent symbols (a jet computation).
gi = matrix(SR, 4, 4, lambda i, j: var("gi%d%d" % (min(i, j), max(i, j))))
A = vector(SR, var("A0 A1 A2 A3")); phi, n, R = var("phi n R")
f, fc = var("f fc"); df = vector(SR, var("df0 df1 df2 df3")); dfc = vector(SR, var("dfc0 dfc1 dfc2 dfc3"))
Au = gi * A
Gi = matrix(SR, 5, 5)
for i in range(4):
    for j in range(4): Gi[i, j] = gi[i, j]
    Gi[i, 4] = Gi[4, i] = -Au[i]
Gi[4, 4] = -phi^(-2) + A * Au
dP = vector(SR, list(df) + [I * n / R * f])          # d_A Phi with the factor e^{iny/R} stripped
dPc = vector(SR, list(dfc) + [-I * n / R * fc])     # d_A Phi^* with e^{-iny/R} stripped
lhs = (dP * Gi * dPc).expand()
D = df - I * n / R * f * A; Dc = dfc + I * n / R * fc * A
rhs = (D * gi * Dc - phi^(-2) * (n / R)^2 * f * fc).expand()
print("[PASS] G^{AB} d_A Phi d_B Phi^* = |(d - i n A/R) f|^2 - phi^{-2} (n/R)^2 |f|^2:", bool((lhs - rhs).expand() == 0))
