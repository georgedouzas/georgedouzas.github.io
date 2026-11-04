# Exercise exr-fbun-hopf-pullback-area: the Hopf map in Euler angles, (theta, phi, psi) -> (theta, phi), pulls the
# area form sin(theta) d theta ^ d phi of S^2 back to minus the differential of the Hopf connection,
# pi^* vol = -d sigma^3 = sigma^1 ^ sigma^2; the curvature of the Hopf connection is the pulled-back area form.
G = Manifold(3, "SU2"); U = G.open_subset("U")
E = U.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi ps:(0,4*pi):\psi"); th, ph, ps = E[:]
S2 = Manifold(2, "S2"); V = S2.open_subset("V")
sph = V.chart(r"t:(0,pi):\theta p:(0,2*pi):\varphi"); t, p = sph[:]
X, Y, Z = sin(t) * cos(p), sin(t) * sin(p), cos(t)
pi_map = U.diff_map(V, {(E, sph): [th, ph]}, name="pi")
vol = V.diff_form(2, name="vol"); vol[0, 1] = sin(t)
pv = pi_map.pullback(vol)
s1 = U.one_form(cos(ps), sin(th) * sin(ps), 0, name="sigma1")
s2 = U.one_form(-sin(ps), sin(th) * cos(ps), 0, name="sigma2")
s3 = U.one_form(0, cos(th), 1, name="sigma3")
print("pi^* vol =", pv.display())
print("d sigma3 =", s3.exterior_derivative().display())
print("pi^* vol = -d sigma3:", pv == -s3.exterior_derivative())
w = s1.wedge(s2); w.apply_map(lambda e: e.simplify_full())
print("pi^* vol = sigma1 ^ sigma2:", pv == w)
print("pi^* vol (d/dpsi, .) = 0, the pulled-back form is horizontal:",
      U.vector_field(0, 0, 1).interior_product(pv).display())
