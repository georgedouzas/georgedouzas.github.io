# Exercise exr-vcp-zero-norm: the normalised graviton zero mode psi(z) = sqrt(k) (k|z| + 1)^{-3/2} has unit norm, and
# the probability of finding it within one AdS length of the brane, |z| < 1/k, is 3/4; within |z| < n/k it is
# 1 - 1/(n + 1)^2. The localisation length is 1/k although z ranges over the whole real line.
z, k, nn = var("z k n", domain="positive")
dens = k * (k * z + 1)**(-3)
print("norm 2 int_0^oo k (kz + 1)^-3 dz =", 2 * integrate(dens, z, 0, oo))
P = (2 * integrate(dens, z, 0, nn / k)).simplify_full()
print("P(|z| < n/k) =", P, "= 1 - 1/(n + 1)^2:", bool((P - (1 - 1 / (nn + 1)**2)).simplify_full() == 0))
print("n = 1, 2, 10:", [P.subs(n=v) for v in [1, 2, 10]])
