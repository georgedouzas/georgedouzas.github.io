# Exercise exr-hso-dft-shift: the discrete circle: the cyclic shift S, the Fourier matrix F and the discrete Laplacian
# L = 2 - S - S^dagger for N = 8 (unitarity, diagonalisation, eigenvalues 4 sin^2(pi k/N), [L, S] = 0), and for N = 64
# the lowest eigenvalues of L/h^2 against n^2 and the heat trace against theta(t).
N = 8
S = matrix(SR, N, N, lambda j, k: 1 if j == (k + 1) % N else 0)
F = matrix(SR, N, N, lambda j, k: exp(2 * pi * I * j * k / N) / sqrt(N))
print("S unitary:", S * S.transpose() == identity_matrix(N), "; F unitary:", bool((F * F.conjugate_transpose()).simplify_full() == identity_matrix(N)))
D = (F.conjugate_transpose() * S * F).simplify_full()
print("F^dagger S F diagonal:", D.is_diagonal(), "; entries:", D.diagonal())
L = 2 * identity_matrix(N) - S - S.transpose()
print("L Hermitian:", L.is_hermitian(), "; [L, S] = 0:", (L * S - S * L).is_zero())
print("eigenvalues of L = 4 sin^2(pi k/N):", sorted([RR(x) for x in L.change_ring(QQ).eigenvalues()]) == sorted([RR(4 * sin(pi * k / N) ** 2) for k in range(N)]) if False else
      all(abs(a - b) < 1e-12 for a, b in zip(sorted(L.change_ring(RDF).eigenvalues()), sorted([float(4 * sin(pi * k / N) ** 2) for k in range(N)]))))
N = 64; h = 2 * pi / N
ev = sorted(float(4 * sin(pi * k / N) ** 2 / h ** 2) for k in range(N))
print("lowest eigenvalues of L/h^2 (N = 64):", [round(x, 4) for x in ev[:7]])
for t in [0.5, 1, 2]:
    print("t = %s: discrete trace %.6f   theta(t) %.6f" % (t, sum(exp(-t * x) for x in ev), sum(exp(-t * n ** 2) for n in range(-40, 41))))
