# Exercise exr-qftx-torus-symmetry: the one-dimensional thermal free energy f = (1/(pi beta^2)) int_0^oo log(1 - e^{-x}) dx
# = -pi/(6 beta^2), the Casimir energy -pi/(6L) on a circle of length L from zeta(-1), and the exchange symmetry of the
# two limits of log Z on the torus.
x = var("x"); b, L = var("beta L", domain="positive")
I0 = integrate(log(1 - exp(-x)), x, 0, oo)
f = I0 / (pi * b ** 2)
E0 = 2 * pi / L * zeta(-1)
print("int_0^oo log(1 - e^{-x}) dx =", I0, "   f(beta) =", f, "   E_0(L) =", E0)
logZ_lowT = -b * E0; logZ_highT = -b * L * f
print("log Z for beta >> L:", logZ_lowT, "   for L >> beta:", logZ_highT)
print("exchanged by beta <-> L:", bool(logZ_lowT.subs({b: L, L: b}) == logZ_highT))
