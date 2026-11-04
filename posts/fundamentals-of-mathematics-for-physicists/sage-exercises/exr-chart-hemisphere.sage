# Exercise exr-chart-hemisphere: the hemispherical chart (X, Y) on the upper half Z > 0 of S^2. Its inverse
# (X, Y) -> (X, Y, sqrt(1 - X^2 - Y^2)) lands on the sphere, its image is the open unit disc, and the south
# stereographic projection (X, Y)/(1 + Z) sends the hemisphere onto the open unit disc of the (u', v')-plane.
X, Y = var("X Y", domain="real")
assume(X**2 + Y**2 < 1)
Z = sqrt(1 - X**2 - Y**2)
print("the inverse chart lands on the sphere, X^2 + Y^2 + Z^2 = 1:", bool((X**2 + Y**2 + Z**2).simplify_full() == 1))
up = X/(1 + Z); vp = Y/(1 + Z)                      # south stereographic coordinates (X, Y)/(1 + Z)
rho2 = (up**2 + vp**2).simplify_full()
print("u'^2 + v'^2 = (1 - Z)/(1 + Z):", bool((rho2 - (1 - Z)/(1 + Z)).simplify_full() == 0))
print("1 - (u'^2 + v'^2) =", (1 - rho2).simplify_full().factor(), " which is positive for Z > 0: the image is inside the unit disc")
J = jacobian((up, vp), (X, Y))
d = J.det().simplify_full()
print("Jacobian determinant of (X, Y) -> (u', v') is 1/(Z (1 + Z)^2):", bool((d - 1/(Z*(1 + Z)**2)).simplify_full() == 0), " (non-zero on the hemisphere)")
print("the centre (X, Y) = (0, 0), the north pole, goes to (u', v') =", (up.subs(X=0, Y=0), vp.subs(X=0, Y=0)))
print("the point (X, Y) = (3/5, 0) with Z = 4/5 goes to (u', v') =", (up.subs(X=3/5, Y=0), vp.subs(X=3/5, Y=0)))
