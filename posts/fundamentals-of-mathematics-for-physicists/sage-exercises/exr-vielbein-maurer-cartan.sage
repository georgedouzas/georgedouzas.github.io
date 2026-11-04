# Exercise exr-vielbein-maurer-cartan: the coset vielbein of S^2 = SU(2)/U(1) as SageManifolds one-forms, e^1 = dtheta,
# e^2 = sin(theta) dphi, e^3 = cos(theta) dphi, and the Maurer-Cartan relations de^3 = -e^1 ^ e^2 (the curvature of the
# U(1) connection e^3) and de^1 = -e^3 ^ e^2 ... with the structure constants of su(2).
S2 = Manifold(2, "S^2"); A = S2.open_subset("A")
sph = A.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi"); th, ph = sph[:]
e1 = A.one_form(1, 0, name="e1"); e2 = A.one_form(0, sin(th), name="e2"); e3 = A.one_form(0, cos(th), name="e3")
print("de^3 =", e3.exterior_derivative().display(), "; -e^1 ^ e^2 =", (-e1.wedge(e2)).display(), "; equal:", e3.exterior_derivative() == -e1.wedge(e2))
print("de^1 =", e1.exterior_derivative().display(), "; -e^2 ^ e^3 =", (-e2.wedge(e3)).display(), "; equal:", e1.exterior_derivative() == -e2.wedge(e3))
print("de^2 =", e2.exterior_derivative().display(), "; -e^3 ^ e^1 =", (-e3.wedge(e1)).display(), "; equal:", e2.exterior_derivative() == -e3.wedge(e1))
g = A.metric("g"); g[0, 0] = 1; g[1, 1] = sin(th)**2
print("e^1 (x) e^1 + e^2 (x) e^2 is the round metric:", (e1 * e1 + e2 * e2) == g)
print("the area form e^1 ^ e^2 =", e1.wedge(e2).display())
