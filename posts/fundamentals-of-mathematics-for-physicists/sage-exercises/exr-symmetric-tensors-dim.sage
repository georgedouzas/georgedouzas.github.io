# Exercise exr-symmetric-tensors-dim: the symmetric tensors in V (x) V (components T^{ij} = T^{ji}) form a subspace of
# dimension n (n + 1) / 2, the antisymmetric ones of dimension n (n - 1) / 2, and the two add up to n^2.
for n in [2, 3, 4]:
    M = MatrixSpace(QQ, n, n)
    sym = span([vector(QQ, (m + m.transpose()).list()) for m in M.basis()], QQ)
    anti = span([vector(QQ, (m - m.transpose()).list()) for m in M.basis()], QQ)
    print("n =", n, ": symmetric", sym.dimension(), "= n(n+1)/2:", sym.dimension() == n * (n + 1) / 2, "; antisymmetric", anti.dimension(), "= n(n-1)/2:", anti.dimension() == n * (n - 1) / 2, "; sum", sym.dimension() + anti.dimension(), "= n^2")
