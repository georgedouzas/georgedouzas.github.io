# Exercise exr-nak-berger-killing: Killing inner products on the squashed (Berger) three-sphere
# g = a^2 ((s1)^2 + (s2)^2) + b^2 (s3)^2 with the left-invariant forms s^a of SU(2) in Euler angles. The right-invariant
# fields xi_a (dual to the forms with ph and ps exchanged) generate left translations and are Killing for all a, b; the
# left-invariant e_3 = d_ps is Killing; e_1 is not unless a = b. The inner products xi_a . xi_b vary over S^3 unless
# a = b, and their average is ((2 a^2 + b^2)/3) delta_ab.
a, b = var("a b"); assume(a > 0, b > 0)
M = Manifold(3, "S3", structure="Riemannian")
X.<th, ph, ps> = M.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi ps:(0,4*pi):\psi")
s = [vector([cos(ps), sin(th) * sin(ps), 0]), vector([-sin(ps), sin(th) * cos(ps), 0]), vector([0, cos(th), 1])]
G = sum(r^2 * v.column() * v.row() for r, v in zip([a, a, b], s))
g = M.metric("g")
for i in range(3):
    for j in range(i, 3): g[i, j] = G[i, j].simplify_full()
# right-invariant forms: exchange ph <-> ps (components in the order th, ph, ps)
sR = [vector([cos(ph), 0, sin(th) * sin(ph)]), vector([-sin(ph), 0, sin(th) * cos(ph)]), vector([0, 1, cos(th)])]
S = matrix(SR, [v.list() for v in s]); SRm = matrix(SR, [v.list() for v in sR])
E = (S.inverse()).apply_map(lambda e: e.simplify_full()); XI = (SRm.inverse()).apply_map(lambda e: e.simplify_full())
e = [M.vector_field(*E.column(k).list()) for k in range(3)]; xi = [M.vector_field(*XI.column(k).list()) for k in range(3)]
iszero = lambda T: all(bool(T[i, j].expr().simplify_full() == 0) for i in range(3) for j in range(3))
print("[PASS] the right-invariant fields xi_a are Killing for every a, b:", all(iszero(g.lie_derivative(v)) for v in xi))
print("[PASS] e_3 = d_ps is Killing, e_1 is not for a != b:", iszero(g.lie_derivative(e[2])) and not iszero(g.lie_derivative(e[0])))
GI = matrix(SR, 3, 3, lambda i, j: (XI.column(i) * G * XI.column(j)).simplify_full())
print("xi_1 . xi_1 =", GI[0, 0].factor())
vol = sin(th)
avg = lambda f: integrate(integrate(integrate(f * vol, th, 0, pi), ph, 0, 2 * pi), ps, 0, 4 * pi) / (16 * pi^2)
A = matrix(SR, 3, 3, lambda i, j: avg(GI[i, j]).simplify_full())
print("[PASS] <xi_a . xi_b> = ((2 a^2 + b^2)/3) delta_ab:", (A - (2 * a^2 + b^2) / 3 * identity_matrix(3)).apply_map(lambda t: t.simplify_full()).is_zero())
print("[PASS] for a = b the inner products are constant, a^2 delta_ab:", (GI.subs(b=a) - a^2 * identity_matrix(3)).apply_map(lambda t: t.simplify_full()).is_zero())
print("[PASS] e_3 . e_3 = b^2 is constant:", bool((E.column(2) * G * E.column(2) - b^2).simplify_full() == 0))
