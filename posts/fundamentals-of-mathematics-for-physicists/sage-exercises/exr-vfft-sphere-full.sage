# Exercise exr-vfft-sphere-full: on the round sphere, the rotation field X = d_phi and the gradient field
# Y = -sin(theta) d_theta: their dual one-forms through the metric, the exterior derivatives of those, the Lie
# derivatives of the metric and of the height along them, their bracket, and Cartan's formula for L_X.
S2 = Manifold(2, "S^2"); A = S2.open_subset("A")
sph = A.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi"); th, ph = sph[:]
g = A.metric("g"); g[0, 0] = 1; g[1, 1] = sin(th)**2
X = A.vector_field(0, 1, name="X"); Y = A.vector_field(-sin(th), 0, name="Y")
Z = A.scalar_field(cos(th), name="Z")
Xb, Yb = X.down(g), Y.down(g)
print("X_flat =", Xb.display(), "; Y_flat =", Yb.display(), "; Y_flat = dZ:", Yb == Z.differential())
print("d(X_flat) =", Xb.exterior_derivative().display(), "; d(Y_flat) =", Yb.exterior_derivative().display())
print("L_X g = 0 (X is a Killing field):", g.lie_derivative(X) == 0)
print("L_Y g =", g.lie_derivative(Y).display(), "; equals -2 cos(theta) g:", g.lie_derivative(Y) == -2*cos(th)*g)
print("L_X Z =", Z.lie_derivative(X).expr(), "; L_Y Z =", Z.lie_derivative(Y).expr())
print("[X, Y] =", X.bracket(Y).display())
al = Xb
cartan = Y.contract(0, al.exterior_derivative(), 0) + al(Y).differential()
print("L_Y(X_flat) =", al.lie_derivative(Y).display())
print("Cartan's formula L_Y alpha = i_Y d alpha + d(alpha(Y)) for alpha = X_flat:", al.lie_derivative(Y) == cartan)
