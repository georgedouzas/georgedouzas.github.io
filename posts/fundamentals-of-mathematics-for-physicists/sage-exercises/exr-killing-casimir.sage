# Exercise exr-killing-casimir: the sum of squares of the three Killing fields of the sphere, acting on functions,
# is the Laplace-Beltrami operator: K_1(K_1 f) + K_2(K_2 f) + K_3(K_3 f) = Delta f; checked on a general function and
# on the spherical harmonic z = cos(theta), an eigenfunction with eigenvalue -2.
M = Manifold(2, "S2", structure="Riemannian"); C = M.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi"); th, ph = C[:]
g = M.metric("g"); g[0, 0] = 1; g[1, 1] = sin(th) ** 2
K = [M.vector_field(-sin(ph), -cos(th) / sin(th) * cos(ph)), M.vector_field(cos(ph), -cos(th) / sin(th) * sin(ph)), M.vector_field(0, 1)]
def casimir(f): return sum(k(k(f)) for k in K)
f = M.scalar_field(function("F")(th, ph), name="F")
diffr = (casimir(f) - f.laplacian(g)).expr().simplify_full()
print("sum_a K_a(K_a F) - Delta F =", diffr)
print("sum_a K_a^i K_a^j = g^{ij}:", all(bool((sum(k[i].expr() * k[j].expr() for k in K) - g.inverse()[i, j].expr()).simplify_full() == 0) for i in range(2) for j in range(2)))
z = M.scalar_field(cos(th), name="z")
print("Casimir on z = cos(theta):", casimir(z).expr().simplify_full(), " = -2 z:", bool((casimir(z).expr() + 2 * cos(th)).simplify_full() == 0))
q = M.scalar_field(3 * cos(th) ** 2 - 1, name="q")
print("Casimir on 3 cos^2(theta) - 1:", bool((casimir(q).expr() + 6 * (3 * cos(th) ** 2 - 1)).simplify_full() == 0), "(eigenvalue -6 = -l(l+1), l = 2)")
