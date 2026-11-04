# Exercise exr-lgt-su2-product: the product of two SU(2) matrices a0 + i a.sigma in components.
a0, a1, a2, a3, b0, b1, b2, b3 = var('a0 a1 a2 a3 b0 b1 b2 b3')
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
q = lambda c0, c: c0 * identity_matrix(SR, 2) + I * sum(ci * si for ci, si in zip(c, s))
a = vector([a1, a2, a3]); b = vector([b1, b2, b3])
c0 = a0*b0 - a.dot_product(b); c = a0*b + b0*a - a.cross_product(b)
print('product formula holds:', (q(a0, a) * q(b0, b) - q(c0, c)).expand().is_zero())
print('c =', c)
print('inverse is (a0, -a) when a0^2 + |a|^2 = 1:', (q(a0, a) * q(a0, -a)).expand() == (a0^2 + a.dot_product(a)) * identity_matrix(SR, 2))
print('determinant:', q(a0, a).det().expand())
