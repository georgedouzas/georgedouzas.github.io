# Exercise exr-vcr-bc-jacobi: the Bianchi normal form [t2, t3] = n1 t1, [t3, t1] = n2 t2 - a t3, [t1, t2] = n3 t3 + a t2
# (Plebanski-Krasinski, Sec. 10.1, eqs. 10.15-10.16). The Jacobi identity holds exactly when a n1 = 0, the traces of the
# adjoint maps are (2a, 0, 0), and the eleven rows of the Bianchi table all satisfy Jacobi; the class A rows (a = 0)
# are the unimodular ones.
a, n1, n2, n3 = var("a n1 n2 n3")
def bracket_table(a, n1, n2, n3):
    f = {}
    def put(i, j, out):
        for c, v in out.items():
            f[(i, j, c)] = v; f[(j, i, c)] = -v
    put(1, 2, {0: n1}); put(2, 0, {1: n2, 2: -a}); put(0, 1, {2: n3, 1: a})
    return lambda i, j, c: f.get((i, j, c), 0)
def jacobi(f):
    return [sum(f(i, j, d) * f(d, k, e) + f(j, k, d) * f(d, i, e) + f(k, i, d) * f(d, j, e) for d in range(3))
            for i in range(3) for j in range(3) for k in range(3) for e in range(3)]
f = bracket_table(a, n1, n2, n3)
print("distinct Jacobi expressions:", sorted(set(expand(x) for x in jacobi(f)), key=str))
print("tr ad(t_1), tr ad(t_2), tr ad(t_3) =", [sum(f(i, b, b) for b in range(3)) for i in range(3)])
# the one-parameter types carry a free a; their invariant is h = a^2/(n2 n3)
table = [("I", 0, 0, 0, 0), ("II", 0, 1, 0, 0), ("VII_0", 0, 1, 1, 0), ("VI_0", 0, 1, -1, 0), ("IX", 0, 1, 1, 1),
         ("VIII", 0, 1, 1, -1), ("V", 1, 0, 0, 0), ("IV", 1, 0, 0, 1), ("VII_h", a, 0, 1, 1), ("VI_h", a, 0, 1, -1),
         ("III", 1, 0, 1, -1)]
for name, A, N1, N2, N3 in table:
    g = bracket_table(A, N1, N2, N3)
    print("%-6s Jacobi %s, traces %s, class %s" % (name, all(expand(x) == 0 for x in jacobi(g)),
          [sum(g(i, b, b) for b in range(3)) for i in range(3)], "A" if bool(A == 0) else "B (a != 0)"))
