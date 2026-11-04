# Exercise exr-killing-minkowski-2d: the Killing vectors of two-dimensional Minkowski space ds^2 = dt^2 - dx^2
# (signature (+,-)): the translations d/dt, d/dx and the boost x d/dt + t d/dx; the rotation-like field
# x d/dt - t d/dx is not Killing. The flow of the boost preserves t^2 - x^2.
M = Manifold(2, "M2", structure="Lorentzian"); C = M.chart("t x"); t, x = C[:]
eta = M.metric("eta", signature=0); eta[0, 0] = 1; eta[1, 1] = -1
fields = {"d/dt": M.vector_field(1, 0), "d/dx": M.vector_field(0, 1), "boost x d/dt + t d/dx": M.vector_field(x, t),
          "x d/dt - t d/dx": M.vector_field(x, -t)}
for name, V in fields.items():
    L = eta.lie_derivative(V)
    print(f"L_({name}) eta =", L.display(), " -> Killing:", L == 0)
w = var("w", domain="real")
tb, xb = t * cosh(w) + x * sinh(w), t * sinh(w) + x * cosh(w)     # flow of the boost by rapidity w
print("boost flow solves d/dw (t, x) = (x, t):", bool(diff(tb, w) == xb and diff(xb, w) == tb))
print("t^2 - x^2 is preserved:", bool((tb ** 2 - xb ** 2 - (t ** 2 - x ** 2)).simplify_full() == 0))
