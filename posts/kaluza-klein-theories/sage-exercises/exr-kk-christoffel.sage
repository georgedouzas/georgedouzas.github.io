# Exercise exr-kk-christoffel: the Christoffel symbols of the ansatz with SageManifolds on an explicit configuration
# (a diagonal four-dimensional metric, the vector A and the scalar phi are polynomials in (t, x) with generic
# rational coefficients; signature (+,-,-,-,-), ansatz g - phi^2 (dy + A)^2), compared with the six patterns of the
# text as exact rational functions:
#   Gamma^l_{mn} = Gamma4^l_{mn} - (phi^2/2)(A_m F_n^l + A_n F_m^l) + phi A_m A_n d^l phi,
#   Gamma^l_{m y} = -(phi^2/2) F_m^l + phi A_m d^l phi,    Gamma^l_{yy} = phi d^l phi,
#   Gamma^y_{mn} = (A_m d_n phi + A_n d_m phi)/phi + (d_m A_n + d_n A_m)/2 - A_l Gamma^l_{mn},
#   Gamma^y_{m y} = d_m phi/phi - A_l Gamma^l_{m y},     Gamma^y_{yy} = -A_l Gamma^l_{yy}.
M5 = Manifold(5, "M5", structure="Lorentzian"); X = M5.chart("t x y z w"); co = X[:]; cv = co[:4]
M4 = Manifold(4, "M4", structure="Lorentzian"); X4 = M4.chart("t x y z"); cv4 = X4[:]
set_random_seed(3)
def rpoly(tt, xx):
    return sum(QQ.random_element(num_bound=4, den_bound=3) * m for m in (1, tt, xx, tt * xx, xx^2, tt^2))
t, x = cv[0], cv[1]
gf = [(1 if i == 0 else -1) * (2 + rpoly(t, x)^2) for i in range(4)]
Af = [rpoly(t, x) for i in range(4)]; phi = 2 + rpoly(t, x)^2
G = M5.metric("G")
for i in range(4):
    for j in range(i, 4): G[i, j] = (gf[i] if i == j else 0) - phi^2 * Af[i] * Af[j]
    G[i, 4] = -phi^2 * Af[i]
G[4, 4] = -phi^2
nab5 = G.connection()
sub4 = dict(zip(cv, cv4))
g4 = M4.metric("g4")
for i in range(4): g4[i, i] = gf[i].subs(sub4)
nab4 = g4.connection(); gi4 = g4.inverse()
A4 = [a.subs(sub4) for a in Af]; phi4 = phi.subs(sub4)
F = lambda m, n: diff(A4[n], cv4[m]) - diff(A4[m], cv4[n])
Fup = lambda m, l: sum(gi4[l, k].expr() * F(m, k) for k in range(4))
dphiup = lambda l: sum(gi4[l, k].expr() * diff(phi4, cv4[k]) for k in range(4))
G5 = lambda k, i, j: nab5[k, i, j].expr().subs(sub4)
Z = lambda e: bool(e.simplify_rational() == 0)
ok = True
for l in range(4):
    for m in range(4):
        for n in range(m, 4):
            ok &= Z(G5(l, m, n) - (nab4[l, m, n].expr() - phi4^2 / 2 * (A4[m] * Fup(n, l) + A4[n] * Fup(m, l)) + phi4 * A4[m] * A4[n] * dphiup(l)))
        ok &= Z(G5(l, m, 4) - (-phi4^2 / 2 * Fup(m, l) + phi4 * A4[m] * dphiup(l)))
    ok &= Z(G5(l, 4, 4) - phi4 * dphiup(l))
print("[PASS] the three patterns Gamma^l_{..} with an upper spacetime index agree with SageManifolds:", ok)
ok2 = True
for m in range(4):
    for n in range(m, 4):
        ok2 &= Z(G5(4, m, n) - ((A4[m] * diff(phi4, cv4[n]) + A4[n] * diff(phi4, cv4[m])) / phi4 + (diff(A4[n], cv4[m]) + diff(A4[m], cv4[n])) / 2 - sum(A4[l] * G5(l, m, n) for l in range(4))))
    ok2 &= Z(G5(4, m, 4) - (diff(phi4, cv4[m]) / phi4 - sum(A4[l] * G5(l, m, 4) for l in range(4))))
ok2 &= Z(G5(4, 4, 4) + sum(A4[l] * G5(l, 4, 4) for l in range(4)))
print("[PASS] the three patterns Gamma^y_{..} agree with SageManifolds:", ok2)
pt = {cv4[0]: QQ(1) / 3, cv4[1]: QQ(-2) / 5, cv4[2]: 0, cv4[3]: 0}
print("example at a point: Gamma^y_{t y} =", G5(4, 0, 4).subs(pt), ", d_t phi / phi - A_l Gamma^l_{t y} =", (diff(phi4, cv4[0]) / phi4 - sum(A4[l] * G5(l, 0, 4) for l in range(4))).subs(pt))
