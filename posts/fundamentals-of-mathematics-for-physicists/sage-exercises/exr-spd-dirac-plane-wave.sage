# Exercise exr-spd-dirac-plane-wave: D = i sigma_a d_a on R^3 acts on e^{ik.x} u as -(k.sigma), with eigenvalues +-|k|,
# and D^2 = -Laplacian on a general spinor field.
x1, x2, x3, k1, k2, k3 = var("x1 x2 x3 k1 k2 k3", domain="real")
X = [x1, x2, x3]
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
D = lambda p: sum(I * s[a] * p.apply_map(lambda f: diff(f, X[a])) for a in range(3))
u1, u2 = var("u1 u2")
pw = exp(I * (k1 * x1 + k2 * x2 + k3 * x3)) * matrix(SR, 2, 1, [u1, u2])
M = -(k1 * s[0] + k2 * s[1] + k3 * s[2])
print("D(e^{ikx} u) = -(k.sigma) e^{ikx} u:", bool((D(pw) - M * pw).simplify_full() == 0))
print("eigenvalues of -(k.sigma):", M.eigenvalues())
f = function("f")(x1, x2, x3); g = function("g")(x1, x2, x3)
psi = matrix(SR, 2, 1, [f, g])
lap = psi.apply_map(lambda h: sum(diff(h, xx, 2) for xx in X))
print("D^2 psi + Laplacian psi =", (D(D(psi)) + lap).simplify_full().list())
