# Exercise exr-lg-killing-so4: so(4) has the basis M_jk = E_jk - E_kj (j < k), dimension 6; its Killing form is
# kappa(X, Y) = (n - 2) tr(XY) = 2 tr(XY), negative definite; compared with so(3), where (n - 2) = 1.
def so_basis(n): return [matrix(QQ, n, n, {(j, k): 1, (k, j): -1}) for j in range(n) for k in range(j + 1, n)]
for n in [3, 4, 5]:
    B = so_basis(n)
    M = matrix(QQ, [b.list() for b in B]).transpose()
    def ad(X): return matrix(QQ, [M.solve_right(vector(QQ, (X * b - b * X).list())) for b in B]).transpose()
    ads = [ad(b) for b in B]
    kap = matrix(QQ, len(B), len(B), lambda a, b: (ads[a] * ads[b]).trace())
    tr = matrix(QQ, len(B), len(B), lambda a, b: (B[a] * B[b]).trace())
    print(f"so({n}): dimension {len(B)}; kappa = {n-2} tr(XY): {kap == (n - 2) * tr}; diagonal of kappa: {kap.diagonal()}; negative definite: {(-kap).is_positive_definite()}")
