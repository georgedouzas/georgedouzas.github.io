# Exercise exr-smooth-atlas-hemisphere-stereographic: the transition map from the hemispherical chart (X, Y) on Z > 0
# to the north stereographic chart, (u, v) = (X, Y)/(1 - sqrt(1 - X^2 - Y^2)) on the upper hemisphere minus N, has
# Jacobian determinant -1/(Z (1 - Z)^2), non-zero there: the hemispherical chart is compatible with the stereographic atlas.
S2 = Manifold(2, "S^2")
H = S2.open_subset("H"); U = S2.open_subset("U")          # upper hemisphere Z > 0; sphere minus N
hem = H.chart("X Y"); ste = U.chart("u v")
X, Y = hem[:]; u, v = ste[:]
Z = sqrt(1 - X**2 - Y**2)
tr = hem.transition_map(ste, (X/(1 - Z), Y/(1 - Z)))        # on H minus N, where 0 < X^2 + Y^2 < 1
tr.set_inverse(2*u/(1 + u**2 + v**2), 2*v/(1 + u**2 + v**2), check=False)
print("hemisphere -> stereographic:", tr.display())
print("stereographic -> hemisphere:", tr.inverse().display())
J = jacobian(tr(X, Y), (X, Y))
d = J.det().simplify_full()
print("Jacobian determinant equals -1/(Z (1 - Z)^2) with Z = sqrt(1 - X^2 - Y^2):", bool((d + 1/(Z*(1 - Z)**2)).simplify_full() == 0))
print("it is finite and non-zero for 0 < Z < 1, that is on the hemisphere minus the pole")
print("the hemisphere corresponds to u^2 + v^2 > 1: at (X, Y) = (3/5, 0), (u, v) =", (tr(X, Y)[0].subs(X=3/5, Y=0), tr(X, Y)[1].subs(X=3/5, Y=0)))
print("and back through the inverse:", [e.subs(u=3, v=0) for e in tr.inverse()(u, v)])
