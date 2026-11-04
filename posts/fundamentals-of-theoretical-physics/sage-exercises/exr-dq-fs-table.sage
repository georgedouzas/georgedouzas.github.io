# Exercise exr-dq-fs-table: offsets (GHz) of the Dirac levels of hydrogen (reduced mass included) from the Bohr levels for n <= 4.
R = RealField(100)
alpha = R(1) / R("137.035999166"); me = R("510998.95069"); h = R("4.135667696e-15"); Mp = R("1836.15267343")
mr = me / (1 + 1 / Mp)
def f(n, j):
    k = R(j) + R(1) / 2
    return (1 + alpha^2 / (n - k + sqrt(k^2 - alpha^2))^2)^(-R(1) / 2)
L = "SPDFG"
for n in range(1, 5):
    bohr = -mr * alpha^2 / (2 * n^2)
    rows = {}
    for l in range(n):
        for j in ([QQ(1) / 2] if l == 0 else [l - QQ(1) / 2, l + QQ(1) / 2]):
            off = (mr * (f(n, j) - 1) - bohr) / h / 1e9
            rows.setdefault(j, []).append("%d%s_%s" % (n, L[l], j))
    print("n = %d:" % n, "; ".join("%s: %.4f GHz" % (", ".join(v), float((mr * (f(n, j) - 1) + mr * alpha^2 / (2 * n^2)) / h / 1e9)) for j, v in sorted(rows.items())))
