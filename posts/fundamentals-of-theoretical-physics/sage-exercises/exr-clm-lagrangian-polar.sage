# Exercise exr-clm-lagrangian-polar: the Lagrangian of a particle in the plane in polar coordinates.
t, m = var("t m")
r, ph = function("r")(t), function("phi")(t)
V = function("V")
x, y = r * cos(ph), r * sin(ph)
L = (m / 2 * (diff(x, t)**2 + diff(y, t)**2)).simplify_full() - V(r)
print("L =", L)
print("L == m/2 (r'^2 + r^2 phi'^2) - V(r):", bool((L - (m / 2 * (diff(r, t)**2 + r**2 * diff(ph, t)**2) - V(r))).simplify_full() == 0))
