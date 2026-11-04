# Exercise exr-fbun-hopf-charge-one: in the Euler parametrisation of SU(2) the fibre coordinate psi has period 4 pi,
# not 2 pi. With the 2 pi-periodic fibre angle chi = psi/2 the Hopf connection is (1/2) sigma^3, its local potentials
# are (1/2)(cos(theta) -+ 1) d phi, the transition function is e^{-i phi} (winding -1) and the first Chern number of
# S^3 -> S^2 is -1; the quotient by psi -> psi + 2 pi, i.e. by (a, b) -> (-a, -b), is SO(3) -> S^2 with c_1 = -2.
th, ph, ps = var("th ph ps", domain="real")
def ab(th, ph, ps):
    return (cos(th / 2) * exp(-I * (ph + ps) / 2), -I * sin(th / 2) * exp(I * (ph - ps) / 2))
a, b = ab(th, ph, ps)
a2, b2 = ab(th, ph, ps + 2 * pi); a4, b4 = ab(th, ph, ps + 4 * pi)
print("psi -> psi + 2 pi sends (a, b) to (-a, -b):", bool((a2 + a).simplify_full() == 0 and (b2 + b).simplify_full() == 0))
print("psi -> psi + 4 pi is the identity:", bool((a4 - a).simplify_full() == 0 and (b4 - b).simplify_full() == 0))
print("the right U(1) action (a, b) -> e^{it}(a, b) is psi -> psi - 2t:",
      bool(all((u - w).simplify_full() == 0 for u, w in zip(ab(th, ph, ps - 2 * var("t")), [e * exp(I * var("t")) for e in (a, b)]))))
S2 = Manifold(2, "S2"); U = S2.open_subset("U")
sph = U.chart(r"t:(0,pi):\theta p:(0,2*pi):\varphi"); t, p = sph[:]
for label, k in [("S^3 -> S^2, fibre angle chi = psi/2", 1 / 2), ("SO(3) -> S^2, fibre angle psi", 1)]:
    Ap = U.one_form(0, k * (cos(t) - 1)); Am = U.one_form(0, k * (cos(t) + 1))
    F = Ap.exterior_derivative()
    c1 = integrate(integrate(F[0, 1].expr(), p, 0, 2 * pi), t, 0, pi) / (2 * pi)
    wind = integrate((Ap - Am)[1].expr(), p, 0, 2 * pi) / (2 * pi)
    print(f"{label}: A_+ = {Ap[1].expr()} dphi, F = {F[0, 1].expr()} dth^dph, c_1 = {c1}, transition winding = {wind}")
