# Exercise exr-fld-int-kk-dirac: a massless Dirac field on M_4 x S^1, Psi = u e^{-i k.x} e^{i n y/R}, with the
# five-dimensional gamma matrices Gamma^mu = gamma^mu, Gamma^4 = i gamma_5, signature (+, -, -, -, -).
# i Gamma^M d_M Psi = 0 becomes D u = 0 with D = gamma^mu k_mu - (n/R) Gamma^4 = gamma^mu k_mu - i (n/R) gamma_5;
# det D vanishes exactly when k^2 = n^2/R^2, and the chiral rotation C = e^{-i pi gamma_5/4} turns D into the
# ordinary Dirac operator C D C = gamma^mu k_mu - n/R: a four-dimensional fermion of mass |n|/R.
w, k3, n, R = var("w k3 n R", domain="real"); assume(R > 0)
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
Z, one = zero_matrix(SR, 2), identity_matrix(SR, 2)
g = [block_matrix([[Z, one], [one, Z]], subdivide=False)] + [block_matrix([[Z, s[i]], [-s[i], Z]], subdivide=False) for i in range(3)]
g5 = I * g[0] * g[1] * g[2] * g[3]; G4 = I * g5; E4 = identity_matrix(SR, 4)
klow = [w, 0, 0, -k3]                                       # k_mu = (omega, 0, 0, -k3)
gk = sum(g[a] * klow[a] for a in range(4))
D = gk - (n / R) * G4
print("D = gamma^mu k_mu - i (n/R) gamma_5:", D == gk - I * (n / R) * g5)
print("det D =", D.det().factor())
print("det D vanishes on w^2 = k3^2 + n^2/R^2:", bool(D.det().subs(w=sqrt(k3**2 + n**2 / R**2)).simplify_full() == 0))
C = (-I * pi / 4 * g5).exp().simplify_full()               # e^{-i pi gamma_5 / 4}
print("C D C = gamma^mu k_mu - n/R:", bool((C * D * C - (gk - (n / R) * E4)).simplify_full() == 0))
print("C gamma^mu C = gamma^mu (gamma_5 anticommutes with gamma^mu):", all(bool((C * g[a] * C - g[a]).simplify_full() == 0) for a in range(4)))
