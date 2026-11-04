# Exercise exr-su2-det-product: the product of two matrices of the SU(2) form has the same form, with
# (a, b)(a', b') = (a a' - conj(b) b', b a' + conj(a) b'), and the determinant is multiplicative: Euler's four-square
# identity |a a' - conj(b) b'|^2 + |b a' + conj(a) b'|^2 = (|a|^2 + |b|^2)(|a'|^2 + |b'|^2).
a1, a2, b1, b2, c1, c2, d1, d2 = var("a1 a2 b1 b2 c1 c2 d1 d2", domain="real")
a = a1 + I*a2; b = b1 + I*b2; ap = c1 + I*c2; bp = d1 + I*d2
def su2(a, b): return matrix(SR, [[a, -conjugate(b)], [b, conjugate(a)]])
g = su2(a, b); gp = su2(ap, bp)
prod = (g*gp).apply_map(lambda e: e.expand())
A = a*ap - conjugate(b)*bp; B = b*ap + conjugate(a)*bp
print("g g' has the SU(2) form with a'' = a a' - conj(b) b', b'' = b a' + conj(a) b':", bool((prod - su2(A, B)).apply_map(lambda e: e.expand()) == 0))
print("det g = |a|^2 + |b|^2:", bool((g.det() - (a1**2 + a2**2 + b1**2 + b2**2)).expand() == 0))
lhs = (A*conjugate(A) + B*conjugate(B)).expand(); rhs = ((a1**2 + a2**2 + b1**2 + b2**2)*(c1**2 + c2**2 + d1**2 + d2**2)).expand()
print("four-square identity |a''|^2 + |b''|^2 = (|a|^2 + |b|^2)(|a'|^2 + |b'|^2):", bool((lhs - rhs).expand() == 0))
g1 = su2(1 + 2*I, 3 - I); g2 = su2(2 + I, 1 + I)
print("example: det g = 15, det g' = 7 and det(g g') =", g1.det().expand(), g2.det().expand(), (g1*g2).det().expand())
print("so matrices of this form with det 1 are closed under products: SU(2) is a group, and (a, b) -> g is the quaternion multiplication")
