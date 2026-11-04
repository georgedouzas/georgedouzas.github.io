# Exercise exr-clb-chain-boundaries (Foulaadvand, Problems 4.1-4.3): normal-mode frequencies of N equal masses joined by
# equal springs (m = k = 1) with fixed-fixed, free-free, periodic and fixed-free ends, by diagonalising the stiffness
# matrix, compared with the closed forms 2 sin(theta/2) for the allowed wave numbers theta.
N = 6
def K(bc):
    A = matrix(QQ, N, N, lambda i, j: 2 if i == j else (-1 if abs(i - j) == 1 else 0))
    if bc in ('free-free',): A[0, 0] = 1; A[N - 1, N - 1] = 1
    if bc == 'fixed-free': A[N - 1, N - 1] = 1
    if bc == 'periodic': A[0, N - 1] = -1; A[N - 1, 0] = -1
    return A
closed = {'fixed-fixed': [2 * sin(p * pi / (2 * (N + 1))) for p in range(1, N + 1)],
          'free-free': [2 * sin(p * pi / (2 * N)) for p in range(0, N)],
          'periodic': [2 * abs(sin(p * pi / N)) for p in range(0, N)],
          'fixed-free': [2 * sin((2 * p - 1) * pi / (2 * (2 * N + 1))) for p in range(1, N + 1)]}
for bc in ('fixed-fixed', 'free-free', 'periodic', 'fixed-free'):
    w = sorted([sqrt(max(e, 0)) for e in K(bc).change_ring(RDF).eigenvalues()])
    c = sorted([RDF(v) for v in closed[bc]])
    print("%-11s omega = %s  matches closed form: %s" % (bc, [round(v, 4) for v in w], all(abs(u - v) < 1e-10 for u, v in zip(w, c))))
