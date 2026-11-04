# Exercise exr-fld-dirac-plane-wave: positive-energy plane waves u e^{-i k.x} of the Dirac equation with momentum
# along z, Weyl representation, signature (+, -, -, -): det(gamma^mu k_mu - m) = (k^2 - m^2)^2, the kernel is
# two-dimensional on the mass shell, and the solutions are u = (sqrt(k.sigma) xi, sqrt(k.sigmabar) xi) with
# k.sigma = omega - p sigma_3, k.sigmabar = omega + p sigma_3. On the shell omega = m cosh(eta), p = m sinh(eta).
w, p, m, eta = var("w p m eta", domain="real"); assume(m > 0)
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
Z, one = zero_matrix(SR, 2), identity_matrix(SR, 2)
g = [block_matrix([[Z, one], [one, Z]], subdivide=False)] + [block_matrix([[Z, s[i]], [-s[i], Z]], subdivide=False) for i in range(3)]
M = lambda w, p: sum(g[a] * [w, 0, 0, -p][a] for a in range(4)) - m * identity_matrix(SR, 4)   # k_mu = (omega, 0, 0, -p)
print("det(gamma^mu k_mu - m) =", M(w, p).det().factor())
W, P = m * (exp(eta) + exp(-eta)) / 2, m * (exp(eta) - exp(-eta)) / 2
Ms = M(W, P)
print("rank on the mass shell:", Ms.simplify_full().rank(), "-> two independent spinors")
rS = sqrt(m) * diagonal_matrix([exp(-eta / 2), exp(eta / 2)])      # square root of k.sigma = diag(omega - p, omega + p)
rSb = sqrt(m) * diagonal_matrix([exp(eta / 2), exp(-eta / 2)])     # square root of k.sigmabar
print("they are the square roots:", bool((rS**2 - (W * one - P * s[2])).simplify_full() == 0), bool((rSb**2 - (W * one + P * s[2])).simplify_full() == 0))
for xi in [vector(SR, [1, 0]), vector(SR, [0, 1])]:
    u = vector(SR, list(rS * xi) + list(rSb * xi))
    print("xi =", xi, ": u =", u, "; (gamma.k - m) u = 0:", bool((Ms * u).simplify_full() == 0))
print("at rest (eta = 0) both blocks equal sqrt(m) xi: gamma^0 u = u")
