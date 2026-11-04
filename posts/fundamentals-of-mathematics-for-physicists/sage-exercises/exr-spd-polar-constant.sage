# Exercise exr-spd-polar-constant: in the polar frame of the plane, psi(theta) = exp(i theta sigma_3 / 2) psi_0 is parallel,
# antiperiodic in its components, and the true holonomy around r = 1 is trivial.
th = var("th"); a, b = var("a b")
s3 = matrix(SR, [[1, 0], [0, -1]])
U = matrix(SR, [[exp(I * th / 2), 0], [0, exp(-I * th / 2)]])
psi = U * vector(SR, [a, b])
nabla = lambda p: vector(SR, [diff(c, th) for c in p]) - I / 2 * s3 * p
print("nabla_theta psi =", nabla(psi).simplify_full())
print("psi(2 pi) = -psi(0):", bool(psi.subs(th=2 * pi) == -psi.subs(th=0)))
print("holonomy (transport times the sign of the antiperiodic frame components):", (-U.subs(th=2 * pi)).simplify_full())
