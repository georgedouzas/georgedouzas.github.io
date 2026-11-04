# Exercise exr-raise-lower-su2: raising and lowering with the SU(2) metric in Euler angles (SageManifolds): the
# cross term mixes the phi and psi components, and down followed by up is the identity on the chart domain.
G = Manifold(3, "SU2", structure="Riemannian"); U = G.open_subset("U")
X = U.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi ps:(0,4*pi):\psi"); th, ph, ps = X[:]; assume(sin(th) > 0)
g = U.metric("g"); g[0, 0] = 1/4; g[1, 1] = 1/4; g[2, 2] = 1/4; g[1, 2] = cos(th)/4
print("g =", g.display()); print("inverse:", g.inverse().display())
dps = U.vector_field(0, 0, 1, name="dpsi"); dph = U.vector_field(0, 1, 0, name="dphi")
print("lowering d_psi:", dps.down(g).display())
print("lowering d_phi:", dph.down(g).display())
dpsi_form = U.one_form(0, 0, 1, name="dpsi")
print("raising the one-form dpsi:", dpsi_form.up(g).display())
print("down then up returns d_psi:", dps.down(g).up(g) == dps)
H = U.vector_field(0, 1, -cos(th), name="H")                       # d_phi - cos(theta) d_psi
print("|d_psi| =", sqrt(g(dps, dps).expr()), "; g(H, d_psi) =", g(H, dps).expr().simplify_full(), "; |H| =", sqrt(g(H, H).expr()).simplify_full())
