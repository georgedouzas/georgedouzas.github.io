# Exercise exr-of-ambient-differentials: the differentials of the three ambient coordinates X, Y, Z restricted to the
# sphere, evaluated on the rotation fields: dX_b(K_a) = K_a(X_b) = -epsilon_abc X_c.
S2 = Manifold(2, "S^2"); A = S2.open_subset("A")
sph = A.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi"); th, ph = sph[:]
K1 = A.vector_field(-sin(ph), -cos(th)/sin(th)*cos(ph), name="K1")
K2 = A.vector_field(cos(ph), -cos(th)/sin(th)*sin(ph), name="K2")
K3 = A.vector_field(0, 1, name="K3")
K = [K1, K2, K3]
Xs = [A.scalar_field(e, name=n) for e, n in [(sin(th)*cos(ph), "X"), (sin(th)*sin(ph), "Y"), (cos(th), "Z")]]
dX = [f.differential() for f in Xs]
for b in range(3):
    print("d%s =" % "XYZ"[b], dX[b].display())
tab = matrix(SR, 3, 3, lambda a, b: dX[b](K[a]).expr().simplify_full())
print("table of dX_b(K_a), rows a = 1, 2, 3, columns X, Y, Z:"); print(tab)
eps = lambda a, b, c: Permutation([a + 1, b + 1, c + 1]).sign() if len({a, b, c}) == 3 else 0
pred = matrix(SR, 3, 3, lambda a, b: -sum(eps(a, b, c)*Xs[c].expr() for c in range(3)))
print("equals -epsilon_abc X_c:", bool((tab - pred).simplify_full() == 0))
