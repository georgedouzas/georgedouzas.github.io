# Exercise exr-flows-killing-conserved: along a geodesic of the round sphere the quantities Q_a = g(K_a, gamma') are
# constant. Checked symbolically: dQ_a/ds, with the second derivatives eliminated by the geodesic equation, equals
# (1/2)(L_{K_a} g)(gamma', gamma') = 0; Q_3 = sin^2(theta) phi' is Clairaut's relation, and Q_a = (x cross x')_a,
# the angular momentum about the a-th axis.
S2 = Manifold(2, "S2", structure="Riemannian"); C = S2.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi"); thc, phc = C[:]
g = S2.metric("g"); g[0, 0] = 1; g[1, 1] = sin(thc) ** 2
Gam = g.christoffel_symbols()
sv = var("s", domain="real")
th = function("th")(sv); ph = function("ph")(sv)
sub = {thc: th, phc: ph}
q = [th, ph]; qd = [diff(th, sv), diff(ph, sv)]
acc = [-sum(Gam[k, i, j].expr().subs(sub) * qd[i] * qd[j] for i in range(2) for j in range(2)) for k in range(2)]
Kc = [[-sin(phc), -cos(thc) / sin(thc) * cos(phc)], [cos(phc), -cos(thc) / sin(thc) * sin(phc)], [0, 1]]
for a in range(3):
    Q = sum(g[i, j].expr().subs(sub) * Kc[a][i].subs(sub) * qd[j] for i in range(2) for j in range(2))
    dQ = diff(Q, sv).subs({diff(th, sv, 2): acc[0], diff(ph, sv, 2): acc[1]})
    print(f"Q_{a+1} = {Q.simplify_full()};  dQ_{a+1}/ds on geodesics = {dQ.simplify_full()}")
X = vector([sin(th) * cos(ph), sin(th) * sin(ph), cos(th)]); Xd = diff(X, sv)
Qs = [sum(g[i, j].expr().subs(sub) * Kc[a][i].subs(sub) * qd[j] for i in range(2) for j in range(2)) for a in range(3)]
print("Q_a = (x cross x')_a for any curve:", bool(all((Qs[a] - X.cross_product(Xd)[a]).simplify_full() == 0 for a in range(3))))
# a concrete geodesic: the great circle in the plane spanned by e1 and (0, cos(al), sin(al)), unit speed
al = pi / 3
Y = vector([cos(sv), sin(sv) * cos(al), sin(sv) * sin(al)]); Yd = diff(Y, sv)
print("great circle with tilt pi/3: Q =", [e.simplify_full() for e in Y.cross_product(Yd)], "(constant, normal to its plane)")
