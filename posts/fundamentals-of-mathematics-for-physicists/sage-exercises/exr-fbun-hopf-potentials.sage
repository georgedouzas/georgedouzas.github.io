# Exercise exr-fbun-hopf-potentials: pulling the Hopf connection sigma^3 = d psi + cos(theta) d phi back by the
# local sections psi = 0, psi = -phi and psi = phi of SU(2) -> S^2 gives cos(theta) d phi and the Dirac potentials
# (cos(theta) - 1) d phi and (cos(theta) + 1) d phi; all three have the same curvature.
G = Manifold(3, "SU2"); U = G.open_subset("U")
E = U.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi ps:(0,4*pi):\psi"); th, ph, ps = E[:]
s3 = U.one_form(0, cos(th), 1, name="sigma3")
S2 = Manifold(2, "S2"); V = S2.open_subset("V")
sph = V.chart(r"t:(0,pi):\theta p:(0,2*pi):\varphi"); t, p = sph[:]
for name, psi_of in [("psi = 0", 0), ("psi = -phi", -p), ("psi = +phi", p)]:
    s = V.diff_map(U, {(sph, E): [t, p, psi_of]}, name="s")
    A = s.pullback(s3)
    print(f"section {name}: A = s^* sigma3 =", A.display(), "; F = dA =", A.exterior_derivative().display())
A0 = V.one_form(0, cos(t)); Am = V.one_form(0, cos(t) - 1)
print("A_- - A_0 = d(-phi) (a gauge transformation):", (Am - A0) == V.scalar_field(-p).differential())
