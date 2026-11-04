# Exercise exr-fbun-chern-charge-n: the monopole of charge n, A_+ = (n/2)(cos(theta) - 1) d phi on the north patch and
# A_- = (n/2)(cos(theta) + 1) d phi on the south patch: c_1 = (1/2 pi) int F = -n, and the same number is the
# winding of the transition function e^{-i n phi} read off from A_+ - A_- on the equator.
S2 = Manifold(2, "S2"); U = S2.open_subset("U")
sph = U.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi"); th, ph = sph[:]
for n in [1, 2, 3, -1]:
    Ap = U.one_form(0, n / 2 * (cos(th) - 1)); Am = U.one_form(0, n / 2 * (cos(th) + 1))
    F = Ap.exterior_derivative()
    c1 = integrate(integrate(F[0, 1].expr(), ph, 0, 2 * pi), th, 0, pi) / (2 * pi)
    wind = integrate((Ap - Am)[1].expr(), ph, 0, 2 * pi) / (2 * pi)
    print(f"n = {n}: F = {F[0, 1].expr()} dth^dph, c_1 = {c1}, winding of the transition = {wind},",
          "same curvature from both patches:", Ap.exterior_derivative() == Am.exterior_derivative())
