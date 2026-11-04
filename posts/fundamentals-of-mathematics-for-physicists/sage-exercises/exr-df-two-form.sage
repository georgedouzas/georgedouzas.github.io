# Exercise exr-df-two-form: the two-form beta = z dx ^ dy + x dy ^ dz on R^3, its components, and its values on the
# pair X = d_x + d_y, Y = d_z in both orders.
M = Manifold(3, "R^3"); cart = M.chart("x y z"); x, y, z = cart[:]
dx, dy, dz = cart.coframe()[:]
beta = z*dx.wedge(dy) + x*dy.wedge(dz); beta.set_name("beta")
print("beta =", beta.display())
print("components beta_ij:"); print(beta[:])
X = M.vector_field(1, 1, 0, name="X"); Y = M.vector_field(0, 0, 1, name="Y")
print("beta(X, Y) =", beta(X, Y).expr(), "; beta(Y, X) =", beta(Y, X).expr(), "; beta(X, X) =", beta(X, X).expr())
