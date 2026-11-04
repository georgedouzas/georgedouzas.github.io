# Exercise exr-dirac-action: the Dirac equation in 1+1 dimensions from the action psi-bar (i gamma^mu d_mu - m) psi with
# 2 x 2 gamma matrices gamma^0 = sigma_1, gamma^1 = i sigma_2 (Clifford relations with eta = diag(1, -1)), varying
# the components of psi-bar as independent jet variables.
t, x, m = var("t x m", domain="real")
g0 = matrix(SR, [[0, 1], [1, 0]]); g1 = I * matrix(SR, [[0, -I], [I, 0]])
eta = diagonal_matrix(SR, [1, -1])
print("Clifford relations in 1+1 dimensions, {gamma^mu, gamma^nu} = 2 eta^{mu nu} with eta = diag(1, -1):", all(g * h + h * g == 2 * eta[a, b] * identity_matrix(SR, 2) for a, g in enumerate([g0, g1]) for b, h in enumerate([g0, g1])))
p1, p2 = function("p1")(t, x), function("p2")(t, x)
psi = vector(SR, [p1, p2])
b1, b2 = var("b1 b2")                                      # the components of psi-bar, treated as independent
psib = vector(SR, [b1, b2])
Dpsi = I * (g0 * diff(psi, t) + g1 * diff(psi, x)) - m * psi
L = psib * Dpsi                                             # L = psi-bar (i gamma^mu d_mu - m) psi
print("varying psi-bar_1 and psi-bar_2 gives the two components of (i gamma^mu d_mu - m) psi = 0:")
print("  ", diff(L, b1).simplify_full(), "= 0")
print("  ", diff(L, b2).simplify_full(), "= 0")
# applying the conjugate operator gives the Klein-Gordon equation on each component
D2 = (I * (g0 * diff(Dpsi, t) + g1 * diff(Dpsi, x)) + m * Dpsi).simplify_full()
print("(i gamma d + m)(i gamma d - m) psi = -(Box + m^2) psi componentwise, Box = d_t^2 - d_x^2:", all(bool((D2[i] + (diff(psi[i], t, 2) - diff(psi[i], x, 2) + m**2 * psi[i])).simplify_full() == 0) for i in range(2)))
