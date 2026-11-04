# Exercise exr-cs-sphere-spin-connection: the spin connection of the sphere is the R-connection. With SageManifolds,
# take the orthonormal frame (E_1, E_2) dual to the coset vielbein e^1 = dtheta, e^2 = sin(theta) dphi, compute the
# Levi-Civita connection one-form omega^1_2 in this frame and its curvature two-form, and compare with -e^3 and
# e^1 ^ e^2.
M = Manifold(2, "S2", structure="Riemannian", start_index=1)
X.<th, ph> = M.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi")
g = M.metric(); g[1, 1], g[2, 2] = 1, sin(th)^2
A = M.automorphism_field(); A[1, 1], A[2, 2] = 1, 1 / sin(th)
E = X.frame().new_frame(A, "E")                  # E_1 = d/dtheta, E_2 = d/dphi / sin(theta)
e = E.coframe()
print("coframe:", e[1].display(), ",", e[2].display())
nab = g.connection()
w = nab.connection_form(1, 2, E)
print("omega^1_2 =", w.display(X.frame()))
e3 = M.one_form(0, cos(th))
print("omega^1_2 = -e^3:", w == -e3)
Om = nab.curvature_form(1, 2, E)
print("Omega^1_2 =", Om.display(X.frame()))
print("Omega^1_2 = e^1 ^ e^2 (Gauss curvature 1):", Om == e[1].wedge(e[2]))
print("d omega^1_2 = Omega^1_2 (abelian in two dimensions):", w.exterior_derivative() == Om)
