# Exercise exr-fan-p-norm-constants: |x|_q <= |x|_p <= n^(1/p - 1/q) |x|_q on R^n, constants for n = 3, and random checks.
n = 3
for p, q in [(1, 2), (2, oo)]:
    c = n^(1 / p - (0 if q == oo else 1 / q))
    x = vector([1, 1, 1])
    print("p = %s, q = %s: constant %s; |x|_p = %s, c |x|_q = %s" % (p, q, c, x.norm(p), c * x.norm(q if q != oo else Infinity)))
set_random_seed(2)
ok = True
for _ in range(500):
    x = vector(RDF, [RDF.random_element(-1, 1) for _ in range(n)])
    for p, q in [(1, 2), (1, 3), (2, 4), (1.5, 2.5)]:
        ok = ok and x.norm(q) <= x.norm(p) + 1e-12 and x.norm(p) <= n^(1 / p - 1 / q) * x.norm(q) + 1e-12
print("both inequalities on 500 random vectors and four pairs (p, q):", ok)
