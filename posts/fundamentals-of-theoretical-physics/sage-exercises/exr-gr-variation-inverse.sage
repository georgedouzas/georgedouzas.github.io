# Exercise exr-gr-variation-inverse: for a general symmetric 3 x 3 matrix g and a symmetric variation dg, the inverse
# varies as d(g^{-1}) = -g^{-1} dg g^{-1}, i.e. delta g^{mu nu} = -g^{mu rho} g^{nu sigma} delta g_{rho sigma}.
eps = var('eps')
a = matrix(SR, 3, 3, lambda i, j: var('g%d%d' % (min(i, j), max(i, j))))
d = matrix(SR, 3, 3, lambda i, j: var('d%d%d' % (min(i, j), max(i, j))))
lhs = diff((a + eps*d).inverse(), eps).subs(eps=0)
rhs = -a.inverse()*d*a.inverse()
print("d/d eps (g + eps dg)^{-1} at eps = 0 equals -g^{-1} dg g^{-1}:", (lhs - rhs).simplify_full() == 0)
