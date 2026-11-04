# Exercise exr-vcr-gm-right-invariant: on the Heisenberg group g = [[1, -m y, z1], [0, 1, z2], [0, 0, 1]] the
# left-invariant fields Z_a (dual to g^{-1} dg) and the right-invariant fields xi_a (dual to dg g^{-1}) agree at the
# identity, commute with each other, and have opposite brackets: [Z_2, Z_y] = m Z_1 but [xi_2, xi_y] = -m xi_1
# (Plebanski-Krasinski, Sec. 10.5, eq. 10.26 and Exercise 3).
y, z1, z2, m = var("y z1 z2 m")
X = [y, z1, z2]
g = matrix(SR, [[1, -m * y, z1], [0, 1, z2], [0, 0, 1]])
gens = [diff(g, v).subs(y=0, z1=0, z2=0) for v in X]          # T_y, T_1, T_2 at the identity
def coeffs(M):                                                 # components of a Lie-algebra matrix in (T_y, T_1, T_2)
    return [M[0, 1] / (-m), M[0, 2], M[1, 2]]
def frame(side):
    rows = []
    for v in X:
        M = g.inverse() * diff(g, v) if side == "left" else diff(g, v) * g.inverse()
        rows.append([simplify(c) for c in coeffs(M)])
    S = matrix(SR, rows).transpose()                           # S[a, mu] = sigma^a_mu
    return S.inverse()                                         # columns: the dual vector fields
def bracket(U, V):
    return vector(SR, [sum(U[j] * diff(V[i], X[j]) - V[j] * diff(U[i], X[j]) for j in range(3)) for i in range(3)]).simplify_full()
L = frame("left"); R = frame("right")
Z = [L.column(a) for a in range(3)]; xi = [R.column(a) for a in range(3)]
names = ["y", "1", "2"]
print("Z  =", [list(v) for v in Z]); print("xi =", [list(v) for v in xi])
print("Z_a(e) = xi_a(e):", all(Z[a].subs(y=0, z1=0, z2=0) == xi[a].subs(y=0, z1=0, z2=0) for a in range(3)))
print("[xi_a, Z_b] = 0 for all a, b:", all(bracket(xi[a], Z[b]) == 0 for a in range(3) for b in range(3)))
print("[Z_2, Z_y] =", list(bracket(Z[2], Z[0])), " m Z_1 =", list(m * Z[1]))
print("[xi_2, xi_y] =", list(bracket(xi[2], xi[0])), " -m xi_1 =", list(-m * xi[1]))
