# Exercise exr-kkr-mass-einstein-frame: the Kaluza-Klein mass in the two frames. For a constant scalar phi the ansatz
# is phi^{-1} gE - phi^2 dy^2 with gE = eta in the Einstein frame, y ~ y + 2 pi R. On Phi = f(t) e^{i n y/R} the
# five-dimensional wave operator gives f'' + m^2 f = 0 with m^2 = n^2/(R^2 phi^3): m = |n|/(R phi^{3/2}) in Einstein-frame
# units, |n|/(R phi) in the Jordan frame g = phi^{-1} eta.
M = Manifold(5, "M", structure="Lorentzian"); X = M.chart("t x v z y"); t, x, v, z, y = X[:]
phi, R, n = var("phi R n"); assume(phi > 0); assume(R > 0)
G = M.metric("G"); G[0, 0] = 1 / phi
for i in (1, 2, 3): G[i, i] = -1 / phi
G[4, 4] = -phi^2
f = function("f")(t)
P = M.scalar_field(f * exp(I * n * y / R))
nab = G.connection()
box = nab(nab(P).up(G)).trace().expr()
red = (box / exp(I * n * y / R)).simplify_full()
print("(Box_5 Phi) e^{-i n y/R} =", red)
print("[PASS] Box_5 Phi = 0 is phi (f'' + n^2/(R^2 phi^3) f) = 0: Einstein-frame mass |n|/(R phi^{3/2}):", bool((red - phi * (diff(f, t, 2) + n^2 / (R^2 * phi^3) * f)).simplify_full() == 0))
print("[PASS] in the Jordan frame g = phi^{-1} eta (proper time t_J = t/sqrt(phi)) m^2 = phi m_E^2 = n^2/(R phi)^2, the mass is |n|/(R phi):", bool((n^2 / (R^2 * phi^3) * phi - n^2 / (R^2 * phi^2)).simplify_full() == 0))
