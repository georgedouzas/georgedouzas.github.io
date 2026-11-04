# Exercise exr-kkr-int-kasner: a cosmology with a shrinking circle. The five-dimensional Kasner metric
# dt^2 - t (dx^2 + dy^2 + dz^2) - t^{-1} dw^2 (exponents p = 1/2, 1/2, 1/2, -1/2 with sum p = sum p^2 = 1) is Ricci
# flat; it is the ansatz with A = 0, phi = t^{-1/2} and the Jordan metric g = dt^2 - t dx^2. The reduced equations
# then require Box phi = 0 and R4 = 0, and the Einstein-frame metric phi g is a flat FRW universe with a(tau) ~ tau^{1/3}
# in its proper time tau, the expansion law of a massless scalar (stiff matter).
M5 = Manifold(5, "M5", structure="Lorentzian"); X5 = M5.chart("t x y z w"); t, x, y, z, w = X5[:]
assume(t > 0)
G = M5.metric("G"); G[0, 0] = 1
for i in (1, 2, 3): G[i, i] = -t
G[4, 4] = -1 / t
print("[PASS] the Kasner metric with exponents (1/2, 1/2, 1/2, -1/2) is Ricci flat:", all(bool(G.ricci()[i, j].expr().simplify_full() == 0) for i in range(5) for j in range(i, 5)))
M = Manifold(4, "M", structure="Lorentzian"); X = M.chart("t x y z"); t, x, y, z = X[:]
g = M.metric("g"); g[0, 0] = 1
for i in (1, 2, 3): g[i, i] = -t
phi = t^(-1 / 2)
nab = g.connection(); box = nab(nab(M.scalar_field(phi)).up(g)).trace().expr().simplify_full()
R4 = g.ricci_scalar().expr().simplify_full()
print("Box phi =", box, "; R4 =", R4)
print("[PASS] Box phi = 0 and R4 = 0, as R_yy = phi Box phi = 0 and R5 = R4 - (2/phi) Box phi = 0 require:", bool(box == 0) and bool(R4 == 0))
tau = var("tau"); assume(tau > 0)
tau_of_t = integrate(t^(-1 / 4), t, 0, t)                  # proper time of gE = phi g: d tau = t^{-1/4} dt
aE2 = phi * t                                             # spatial scale factor squared of gE: t^{1/2}
t_of_tau = (3 * tau / 4)^(4 / 3)                         # inverse of tau = (4/3) t^{3/4}
assert bool((tau_of_t.subs(t=t_of_tau) - tau).simplify_full() == 0)
aE = (t_of_tau)^(1 / 4)                                   # a_E = sqrt(phi t) = t^{1/4}
print("tau(t) =", tau_of_t, "; a_E(tau)^3 =", (aE^3).canonicalize_radical())
print("[PASS] the Einstein-frame scale factor grows as tau^{1/3}: a_E = (3 tau/4)^{1/3}:", bool((aE^12 - (3 * tau / 4)^4).canonicalize_radical() == 0) and bool((aE2 - sqrt(t)).simplify_full() == 0))
