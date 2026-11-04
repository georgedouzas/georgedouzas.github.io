# Exercise exr-fan-volterra-singular-values: singular values of the Volterra operator V f(x) = int_0^x f on L^2(0, 1),
# exact values 2/((2k - 1) pi), compared with the singular values of the discretisation h * (1 if i >= j else 0).
N = 200
h = 1.0 / N
M = matrix(RDF, N, N, lambda i, j: h if i >= j else 0.0)
sv = sorted(M.singular_values(), reverse=True)
exact = [2 / ((2 * k - 1) * pi.n()) for k in range(1, 7)]
print("k   discrete s_k    exact 2/((2k-1)pi)")
for k in range(6):
    print(k + 1, " %.6f" % sv[k], "  %.6f" % exact[k])
print("relative error of s_1:", abs(sv[0] - exact[0]) / exact[0])
