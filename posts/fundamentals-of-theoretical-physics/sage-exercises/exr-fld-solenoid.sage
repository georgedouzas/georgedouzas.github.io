# Exercise exr-fld-solenoid: the vector potential of an infinitely thin solenoid of flux Phi along the z-axis,
# A = (Phi / (2 pi r)) e_phi, i.e. the spatial one-form (Phi/(2 pi)) dphi. Its curl vanishes for r > 0, yet its line
# integral around the axis is Phi: a field strength zero outside, a potential that cannot be gauged away globally.
x, y, z, P = var("x y z P", domain="real")
r2 = x**2 + y**2
A = vector([-P * y / (2 * pi * r2), P * x / (2 * pi * r2), 0])
curl = vector([diff(A[2], y) - diff(A[1], z), diff(A[0], z) - diff(A[2], x), diff(A[1], x) - diff(A[0], y)])
print("B = curl A for r > 0:", curl.simplify_full())
s, R = var("s R"); assume(R > 0)
c = vector([R * cos(s), R * sin(s), 0])
lineint = integrate((A.subs(x=c[0], y=c[1]) * diff(c, s)).simplify_full(), s, 0, 2 * pi)
print("line integral of A around a circle of radius R:", lineint)
lam = P * atan2(y, x) / (2 * pi)
print("locally A = grad(Phi phi/(2 pi)):", (vector([diff(lam, x), diff(lam, y), 0]) - A).simplify_full())
