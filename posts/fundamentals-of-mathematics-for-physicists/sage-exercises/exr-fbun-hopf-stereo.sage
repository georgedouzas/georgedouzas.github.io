# Exercise exr-fbun-hopf-stereo: composing the Hopf map with stereographic projection from the south pole,
# (X + iY)/(1 + Z), gives conj(b)/conj(a): the point of S^2 is the ratio of the two coordinates, so the Hopf map is
# the map S^3 -> CP^1, (a, b) -> [a : b], and it is visibly constant on (a e^{it}, b e^{it}).
a1, a2, b1, b2, t = var("a1 a2 b1 b2 t", domain="real")
a, b = a1 + I * a2, b1 + I * b2
ab = (a * b.conjugate()).expand()
X, Y, Z = 2 * ab.real(), 2 * ab.imag(), a1 ** 2 + a2 ** 2 - b1 ** 2 - b2 ** 2
on_sphere = {b2: sqrt(1 - a1 ** 2 - a2 ** 2 - b1 ** 2)}
lhs = ((X + I * Y) / (1 + Z)).subs(on_sphere)
rhs = (b.conjugate() / a.conjugate()).subs(on_sphere)
print("(X + iY)/(1 + Z) = conj(b)/conj(a) on S^3:", bool((lhs - rhs).simplify_full() == 0))
print("X^2 + Y^2 + Z^2 = (|a|^2 + |b|^2)^2:", bool((X ** 2 + Y ** 2 + Z ** 2 - (a1**2 + a2**2 + b1**2 + b2**2) ** 2).expand() == 0))
ra, rb = (a * exp(I * t)), (b * exp(I * t))
print("the ratio conj(b)/conj(a) is unchanged by (a, b) -> e^{it}(a, b):",
      bool((rb.conjugate() / ra.conjugate() - b.conjugate() / a.conjugate()).simplify_full() == 0))
print("the north pole Z = 1 is b = 0 (ratio 0); the south pole Z = -1 is a = 0 (ratio infinite)")
