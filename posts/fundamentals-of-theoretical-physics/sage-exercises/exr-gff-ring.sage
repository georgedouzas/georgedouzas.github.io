# Exercise exr-gff-ring (Bauerschmidt-Brydges-Slade, Def. 1.5.1, eqs. 1.5.3-1.5.4, on a ring): the Gaussian free field on
# a ring of L = 40 sites with m^2 = 1/10. Covariance (-Delta + m^2)^{-1} by exact inversion against the image sum
# cosh((L/2 - x)/xi)/(2 sinh(1/xi) sinh(L/(2 xi))), cosh(1/xi) = 1 + m^2/2; susceptibility sum_x C_0x = 1/m^2 exactly;
# free energy per site (1/(2L)) log det(-Delta + m^2) against (1/2) arccosh(1 + m^2/2).
L = 40; m2 = QQ(1) / 10
A = matrix(QQ, L, L, lambda i, j: 1 if (i - j) % L in (1, L - 1) else 0)
M = (2 + m2) * identity_matrix(QQ, L) - A
C = M.inverse()
print("sum_x C_0x =", sum(C[0, x] for x in range(L)), " (1/m^2 =", 1 / m2, ")")
xi = 1 / arccosh(1 + RDF(m2) / 2)
img = lambda x: cosh((L / 2 - x) / xi) / (2 * sinh(1 / xi) * sinh(L / (2 * xi)))
print("xi = %.6f, 1/m = %.6f" % (xi, 1 / sqrt(RDF(m2))))
print("max |C_0x - image sum| over x:", max(abs(RDF(C[0, x]) - img(x)) for x in range(L)))
for x in (0, 1, 5, 10, 20):
    print("x = %2d: C_0x = %.8f" % (x, RDF(C[0, x])))
print("(1/(2L)) log det = %.8f, (1/2) arccosh(1 + m^2/2) = %.8f" % (RDF(log(M.det())) / (2 * L), 1 / (2 * xi)))
