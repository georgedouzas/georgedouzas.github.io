# Exercise exr-sr-reverse-cs-components: the identity behind the reverse Cauchy-Schwarz inequality, and a random test of
# u.w >= |u||w| and |u + w| >= |u| + |w| for future-directed timelike vectors.
var('a b c d')
print("(ac - bd)^2 - (a^2 - b^2)(c^2 - d^2) =", ((a*c - b*d)^2 - (a^2 - b^2)*(c^2 - d^2)).factor())
set_random_seed(2)
eta = diagonal_matrix(RDF, [1, -1, -1, -1])
def rand_future():
    s = vector(RDF, [RDF.random_element(-1, 1) for _ in range(3)])
    return vector(RDF, [s.norm() + RDF.random_element(0.01, 1)] + list(s))
worst_cs, worst_tri = 10.0, 10.0
for _ in range(2000):
    u, w = rand_future(), rand_future()
    nu, nw = sqrt(u*eta*u), sqrt(w*eta*w)
    worst_cs = min(worst_cs, u*eta*w - nu*nw)
    worst_tri = min(worst_tri, sqrt((u + w)*eta*(u + w)) - nu - nw)
print("min of u.w - |u||w| over 2000 pairs: %.3e (>= 0)" % worst_cs)
print("min of |u+w| - |u| - |w| over 2000 pairs: %.3e (>= 0)" % worst_tri)
