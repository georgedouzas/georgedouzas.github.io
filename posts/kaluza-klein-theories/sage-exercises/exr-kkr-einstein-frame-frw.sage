# Exercise exr-kkr-einstein-frame-frw: the Einstein-frame identity on a cosmological metric. With
# gE = dt^2 - a(t)^2 (dx^2 + dy^2 + dz^2) and phi = phi(t), the Jordan metric g = phi^{-1} gE satisfies
# sqrt(-g) phi R[g] = sqrt(-gE) (R_E + 3 Box_E ln phi - (3/2) gE^{mu nu} d_mu ln phi d_nu ln phi), checked after
# dividing both sides by sqrt(-gE) = a^3 (sqrt(-g) = phi^{-2} a^3).
M = Manifold(4, "M", structure="Lorentzian"); X = M.chart("t x y z"); t, x, y, z = X[:]
a = function("a")(t); phi = function("phi")(t)
gE = M.metric("gE"); gE[0, 0] = 1
for i in (1, 2, 3): gE[i, i] = -a^2
g = M.metric("g")
for i in range(4): g[i, i] = gE[i, i].expr() / phi
RE = gE.ricci_scalar().expr(); R = g.ricci_scalar().expr()
nab = gE.connection(); L = M.scalar_field(log(phi))
boxL = nab(nab(L).up(gE)).trace().expr()
dL2 = diff(log(phi), t)^2                                # gE^{tt} = 1
print("R_E =", RE.simplify_full())
print("[PASS] phi^{-1} R[g] = R_E + 3 Box_E ln phi - (3/2)(d ln phi)^2 on the FRW metric:", bool((R / phi - (RE + 3 * boxL - 3 / 2 * dL2)).simplify_full() == 0))
