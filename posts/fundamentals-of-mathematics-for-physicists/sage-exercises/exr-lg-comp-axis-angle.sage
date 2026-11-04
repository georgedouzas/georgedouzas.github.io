# Exercise exr-lg-comp-axis-angle: compose the quarter turns about e_1 and e_2 in SU(2), read off the angle from the
# trace and the axis from tr(sigma_a g) and from an eigenvector, and check that Ad(g) is the product of the rotations.
sig = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
T = [-I / 2 * s for s in sig]
L = [matrix(SR, [[0, 0, 0], [0, 0, -1], [0, 1, 0]]), matrix(SR, [[0, 0, 1], [0, 0, 0], [-1, 0, 0]]), matrix(SR, [[0, -1, 0], [1, 0, 0], [0, 0, 0]])]
simp = lambda M: M.apply_map(lambda e: e.simplify_full())
def g(t, n):  # e^{t n.T} = cos(t/2) 1 - i sin(t/2) n.sigma
    return cos(t / 2) * identity_matrix(SR, 2) - I * sin(t / 2) * sum(n[a] * sig[a] for a in range(3))
def rot(t, n):  # Rodrigues' formula
    N = sum(n[a] * L[a] for a in range(3))
    return simp(identity_matrix(SR, 3) + sin(t) * N + (1 - cos(t)) * N^2)
def Ad(h):  # matrix of X -> h X h^{-1} in the basis T_a: the component along T_a is -2 tr(T_a .)
    return simp(matrix(SR, 3, 3, lambda a, b: -2 * (T[a] * h * T[b] * h.inverse()).trace()))
g1, g2 = g(pi / 2, [1, 0, 0]), g(pi / 2, [0, 1, 0])
G = simp(g1 * g2)
c = (G.trace() / 2).simplify_full()
s_n = vector(SR, [(I / 2 * (sig[a] * G).trace()).simplify_full() for a in range(3)])
t = 2 * arccos(c); n = (s_n / sin(t / 2)).apply_map(lambda e: e.simplify_full())
print("cos(t/2) =", c, "; sin(t/2) n =", s_n, "; angle t =", t, "; axis n =", n)
ev = G.change_ring(QQbar).eigenvectors_right()
print("eigenvalues of g:", [e.radical_expression() for (e, _, _) in ev], "; e^{-i t/2} =", exp(-I * t / 2).rectform())
xi = [vector(QQbar, list(vs[0])) for (e, vs, _) in ev if e == QQbar(exp(-I * pi / 3))][0]
xi = xi / sqrt(sum(abs(x)^2 for x in xi))
print("axis xi^dagger sigma xi from the eigenvector of e^{-i t/2}:", [(xi.conjugate() * sig[a].change_ring(QQbar) * xi).radical_expression() for a in range(3)])
R12 = simp(rot(pi / 2, [1, 0, 0]) * rot(pi / 2, [0, 1, 0]))
print("R_1(pi/2) R_2(pi/2) =", R12.list(), "(e_1 -> e_2 -> e_3 -> e_1)")
print("Ad(g1 g2) = R_1(pi/2) R_2(pi/2):", Ad(G) == R12, "; = rotation by 2 pi/3 about (1,1,1)/sqrt3:", R12 == rot(2 * pi / 3, [1 / sqrt(3)] * 3))
print("reversed order, Ad(g2 g1) = rotation by 2 pi/3 about (1,1,-1)/sqrt3:", Ad(simp(g2 * g1)) == rot(2 * pi / 3, [1 / sqrt(3), 1 / sqrt(3), -1 / sqrt(3)]))
