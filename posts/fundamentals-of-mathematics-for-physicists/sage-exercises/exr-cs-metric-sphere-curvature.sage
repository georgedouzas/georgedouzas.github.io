# Exercise exr-cs-metric-sphere-curvature: the invariant metric of S^2 = SO(3)/SO(2) built from the coset vielbein,
# ds^2 = R_0^2 ((e^1)^2 + (e^2)^2), with SageManifolds: its Ricci scalar is 2/R_0^2, and the three Killing fields of the
# sphere preserve it (it is SO(3)-invariant).
R0 = var("R0"); assume(R0 > 0)
M = Manifold(2, "S2", structure="Riemannian")
X.<th, ph> = M.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi")
e1 = M.one_form(R0, 0, name="e1"); e2 = M.one_form(0, R0 * sin(th), name="e2")
g = M.metric(); g.set(e1 * e1 + e2 * e2)
print("g =", g.display())
print("Ricci scalar =", g.ricci_scalar().expr().simplify_full())
K = [M.vector_field(-sin(ph), -cos(th) / sin(th) * cos(ph)), M.vector_field(cos(ph), -cos(th) / sin(th) * sin(ph)), M.vector_field(0, 1)]
print("L_K g = 0 for the three Killing fields K_1, K_2, K_3:", all(g.lie_derivative(k) == 0 for k in K))
