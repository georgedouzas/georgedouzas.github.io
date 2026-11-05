# Oracle: the abelian Kaluza-Klein reduction checked with SageManifolds on an explicit configuration. The 4D metric
# (diagonal), the vector A and the scalar phi are explicit polynomials in (t, x) with generic rational coefficients, so
# the jets along t and x take generic values; the five-dimensional curvature scalar of ds^2 = g - phi^2 (dw + A)^2 (signature
# (+,-,-,-,-)) is computed from the Levi-Civita connection and compared with R4 + (phi^2/4) F^2 - (2/phi) Box phi, together
# with sqrt(-g5) = phi sqrt(-g4), i.e. det g5 = -phi^2 det g4.
# The identity holds exactly, as rational functions. Jet-polynomial form of the same identity: sage/kk_jets.sage.
M5 = Manifold(5, "M5", structure="Lorentzian"); X5 = M5.chart("t x y z w"); t, x, y, z, w = X5[:]
M4 = Manifold(4, "M4", structure="Lorentzian"); X4 = M4.chart("t x y z"); t4, x4, y4, z4 = X4[:]
set_random_seed(7)
def rpoly(cv):
    tt, xx, yy, zz = cv
    mons = [1, tt, xx, tt*xx, xx^2, tt^2]
    return sum(QQ.random_element(num_bound=5, den_bound=3) * m for m in mons)
cv5 = (t, x, y, z); cv4 = (t4, x4, y4, z4)
gpoly = {}
for i in range(4):
    for j in range(i, 4):
        gpoly[(i, j)] = 0 if i != j else ((1 if i == 0 else -1) * (3 + rpoly(cv5)^2))     # diagonal, Lorentzian (+,-,-,-)
Apoly = [rpoly(cv5) for i in range(4)]; phipoly = 2 + rpoly(cv5)^2
sub4 = {t: t4, x: x4, y: y4, z: z4}
g5 = M5.metric("g5"); g4 = M4.metric("g4")
for i in range(4):
    for j in range(i, 4):
        g5[i, j] = gpoly[(i, j)] - phipoly^2 * Apoly[i] * Apoly[j]
        g4[i, j] = gpoly[(i, j)].subs(sub4)
    g5[i, 4] = -phipoly^2 * Apoly[i]
g5[4, 4] = -phipoly^2
print("computing the five-dimensional curvature scalar of the ansatz metric ...")
R5 = g5.ricci_scalar().expr()
R4 = g4.ricci_scalar().expr()
phi4 = phipoly.subs(sub4); A4 = [a.subs(sub4) for a in Apoly]
ginv4 = g4.inverse()
F = [[diff(A4[n], cv4[m]) - diff(A4[m], cv4[n]) for n in range(4)] for m in range(4)]
F2 = sum(ginv4[m, k].expr() * ginv4[n, l].expr() * F[m][n] * F[k][l] for m in range(4) for n in range(4) for k in range(4) for l in range(4))
nab = g4.connection()
box = sum(ginv4[m, n].expr() * (diff(phi4, cv4[m], cv4[n]) - sum(nab[k, m, n].expr() * diff(phi4, cv4[k]) for k in range(4))) for m in range(4) for n in range(4))
target = R4 + phi4^2 / 4 * F2 - 2 / phi4 * box
resid = (R5.subs(sub4) - target)
print("[PASS] R5 = R4 + (phi^2/4) F^2 - (2/phi) Box phi on the explicit configuration (exact rational functions):", bool(resid.simplify_rational() == 0))
det5 = g5.determinant().expr(); det4 = g4.determinant().expr()
print("[PASS] det g5 = -phi^2 det g4 on the explicit configuration:", bool((det5.subs(sub4) + phi4^2 * det4).simplify_rational() == 0))
# a numerical spot check at a point, to show the sizes involved
pt = {t4: QQ(1)/3, x4: QQ(-1)/2, y4: QQ(2)/5, z4: QQ(1)/7}
print("at a sample point: R5 =", R5.subs(sub4).subs(pt), ", R4 =", R4.subs(pt), ", (phi^2/4)F^2 =", (phi4^2 / 4 * F2).subs(pt), ", (2/phi) Box phi =", (2 / phi4 * box).subs(pt))
