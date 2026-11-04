# Exercise exr-kk-gauge-shift: the coordinate change y' = y + eps(x) in five dimensions. With SageManifolds, write the
# ansatz metric ds^2 = g - phi^2 (dw + A)^2 in the new chart and read off the transformed fields: g and phi unchanged,
# A' = A - d eps.
M = Manifold(5, "M", structure="Lorentzian"); X = M.chart("t x y z w"); t, x, y, z, w = X[:]
Xp = M.chart("tp xp yp zp wp"); tp, xp, yp, zp, wp = Xp[:]
eps = function("eps")(t, x, y, z)
tr = X.transition_map(Xp, [t, x, y, z, w + eps]); tr.set_inverse(tp, xp, yp, zp, wp - eps.subs({t: tp, x: xp, y: yp, z: zp}))
gf = [function("g%d" % i)(t, x, y, z) for i in range(4)]; Af = [function("A%d" % i)(t, x, y, z) for i in range(4)]; phi = function("phi")(t, x, y, z)
G = M.metric("G")
for i in range(4):
    for j in range(i, 4): G[i, j] = (gf[i] if i == j else 0) - phi^2 * Af[i] * Af[j]
    G[i, 4] = -phi^2 * Af[i]
G[4, 4] = -phi^2
comp = G.comp(Xp.frame())
sub = {tp: t, xp: x, yp: y, zp: z}
Ap = [Af[i] - diff(eps, [t, x, y, z][i]) for i in range(4)]
ok = True
for i in range(4):
    for j in range(i, 4): ok &= bool((comp[i, j, Xp].expr().subs(sub) - ((gf[i] if i == j else 0) - phi^2 * Ap[i] * Ap[j])).simplify_full() == 0)
    ok &= bool((comp[i, 4, Xp].expr().subs(sub) + phi^2 * Ap[i]).simplify_full() == 0)
ok &= bool((comp[4, 4, Xp].expr().subs(sub) + phi^2).simplify_full() == 0)
print("G_{t w} in the shifted chart:", comp[0, 4, Xp].expr().subs(sub).simplify_full())
print("[PASS] under w' = w + eps(x) the ansatz keeps its form with g' = g, phi' = phi and A' = A - d eps:", ok)
