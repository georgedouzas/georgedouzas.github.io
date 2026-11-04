# Exercise exr-tf-components: the type (1, 1) tensor field T = d_phi (x) dtheta + sin(theta) d_theta (x) dphi on the
# sphere: its components T^i_j = T(dx^i, d_j), and the vector field T(X) obtained by contracting with X.
S2 = Manifold(2, "S^2"); A = S2.open_subset("A")
sph = A.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi"); th, ph = sph[:]
dth, dph = sph.coframe()[0], sph.coframe()[1]
e_th, e_ph = sph.frame()[0], sph.frame()[1]
T = e_ph*dth + sin(th)*e_th*dph
print("T =", T.display(), "; type", T.tensor_type())
print("component matrix T^i_j (row i, column j):"); print(T[:])
print("T(dphi, d_theta) =", T(dph, e_th).expr(), "; T(dtheta, d_phi) =", T(dth, e_ph).expr(), "; T(dtheta, d_theta) =", T(dth, e_th).expr())
X = A.vector_field(1, 2, name="X")
print("T(X) for X = d_theta + 2 d_phi:", T.contract(1, X, 0).display())
print("trace T^i_i =", T.trace().expr())
