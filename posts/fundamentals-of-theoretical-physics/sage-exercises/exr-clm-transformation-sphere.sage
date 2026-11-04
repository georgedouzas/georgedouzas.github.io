# Exercise exr-clm-transformation-sphere: the rotation about the X-axis, delta(X, Y, Z) = (0, -Z, Y), in spherical
# coordinates: delta theta and delta phi from theta = arccos Z, phi = atan2(Y, X).
th, ph = var("theta phi", domain="real"); X, Y, Z = var("X Y Z", domain="real")
assume(0 < th, th < pi)
thf = arccos(Z); phf = arctan2(Y, X)
dX = vector([0, -Z, Y])
pt = {X: sin(th) * cos(ph), Y: sin(th) * sin(ph), Z: cos(th)}
dth = sum(diff(thf, v) * c for v, c in zip([X, Y, Z], dX)).subs(pt).simplify_full().subs({abs(sin(th)): sin(th)})  # sin(theta) > 0 on the chart
dph = sum(diff(phf, v) * c for v, c in zip([X, Y, Z], dX)).subs(pt).simplify_full()
print("delta theta =", dth, "; delta phi =", dph)
print("delta theta == -sin(phi):", bool((dth + sin(ph)).simplify_full() == 0), "; delta phi == -cot(theta) cos(phi):", bool((dph + cot(th) * cos(ph)).simplify_full() == 0))
