# Exercise exr-lie-derivative-oneform: Lie derivatives of one-forms with SageManifolds, and the identities
# L_X df = d(X f) and (L_X w)(Y) = X(w(Y)) - w([X, Y]) on the plane.
M = Manifold(2, "R^2"); C = M.chart("x y"); x, y = C[:]
rot = M.vector_field(-y, x, name="rot"); rad = M.vector_field(x, y, name="rad")
f = M.scalar_field(x ** 2 * y, name="f")
for name, V in [("rot", rot), ("rad", rad)]:
    w = f.differential()
    print(f"L_{name} df =", w.lie_derivative(V).display(), f"  equals d({name}(f)):", w.lie_derivative(V) == V(f).differential())
dx = C.coframe()[0]; dy = C.coframe()[1]
print("L_rot dx =", dx.lie_derivative(rot).display(), ";  L_rot dy =", dy.lie_derivative(rot).display())
print("L_rad dx =", dx.lie_derivative(rad).display(), ";  L_rad dy =", dy.lie_derivative(rad).display())
w = M.one_form(y ** 2, x * y, name="w"); Y = M.vector_field(1, x, name="Y")
lhs = w.lie_derivative(rot)(Y); rhs = rot(w(Y)) - w(rot.bracket(Y))
print("(L_rot w)(Y) = rot(w(Y)) - w([rot, Y]):", lhs == rhs)
