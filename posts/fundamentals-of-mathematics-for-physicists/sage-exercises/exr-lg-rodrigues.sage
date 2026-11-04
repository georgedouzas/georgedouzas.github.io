# Exercise exr-lg-rodrigues: (n.L)^3 = -|n|^2 (n.L), hence for a unit vector n the exponential series sums to
# Rodrigues' formula e^{t n.L} = 1 + sin t (n.L) + (1 - cos t)(n.L)^2; for n = (1,1,1)/sqrt 3 and t = 2 pi/3 it is
# the cyclic permutation of the axes.
L = [matrix(SR, [[0, 0, 0], [0, 0, -1], [0, 1, 0]]), matrix(SR, [[0, 0, 1], [0, 0, 0], [-1, 0, 0]]), matrix(SR, [[0, -1, 0], [1, 0, 0], [0, 0, 0]])]
n = vector(SR, var("n1 n2 n3", domain="real"))
N = sum(n[a] * L[a] for a in range(3))
print("(n.L)^3 + |n|^2 (n.L) =", (N^3 + n.dot_product(n) * N).expand().list())
print("(n.L)^2 = n n^T - |n|^2 1:", bool((N^2).expand() == (n.column() * n.row() - n.dot_product(n) * identity_matrix(SR, 3)).expand()))
t = var("t", domain="real")
u = vector(SR, [1, 1, 1]) / sqrt(3); U = sum(u[a] * L[a] for a in range(3))
rod = identity_matrix(SR, 3) + sin(t) * U + (1 - cos(t)) * U^2
ex = (t * U).exp().apply_map(lambda e: e.rectform().simplify_full())
print("exp(t u.L) - Rodrigues =", (ex - rod).apply_map(lambda e: e.simplify_full().trig_reduce().simplify_full()).list())
P = rod.subs(t=2 * pi / 3).simplify_full()
print("rotation by 2 pi/3 about (1,1,1):", P.list(), "; sends e1 -> e2 -> e3 -> e1:", P * vector([1, 0, 0]) == vector([0, 1, 0]) and P * vector([0, 1, 0]) == vector([0, 0, 1]))
