# Exercise exr-mexp-sage: matrix exponentials of the nilpotent N, of diag(1, -1) and of M = [[1, 1], [0, 1]], and the
# derivative identity d/dt exp(tM) = M exp(tM).
t = var("t")
N = matrix(QQ, [[0, 1], [0, 0]])
D = matrix(QQ, [[1, 0], [0, -1]])
M = matrix(QQ, [[1, 1], [0, 1]])
for name, X in [("N", N), ("diag(1, -1)", D), ("M", M)]:
    E = (t * X).exp().apply_map(lambda e: e.simplify_full())
    print("exp(t", name, ") =", E.list())
EM = (t * M).exp().apply_map(lambda e: e.simplify_full())
print("exp(tM) = e^t (1 + tN):", bool((EM - exp(t) * (identity_matrix(2) + t * N)).apply_map(lambda e: e.simplify_full()) == 0))
print("d/dt exp(tM) = M exp(tM):", bool((EM.derivative(t) - M * EM).apply_map(lambda e: e.simplify_full()) == 0))
