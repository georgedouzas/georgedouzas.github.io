# Exercise exr-vcr-unimod-sage: the trace of the adjoint map, tr ad(t_a) = f_ab^b, for four three-dimensional Lie
# algebras given by their brackets: so(3), sl(2, R), the Heisenberg algebra, and Bianchi V ([t3, t1] = t1, [t3, t2] = t2).
def structure(brackets, n=3):
    f = {}
    for (a, b), out in brackets.items():
        for c, v in out.items():
            f[(a, b, c)] = v; f[(b, a, c)] = -v
    return lambda a, b, c: f.get((a, b, c), 0)
algs = {
    "so(3)": structure({(0, 1): {2: 1}, (1, 2): {0: 1}, (2, 0): {1: 1}}),
    "sl(2,R)": structure({(0, 1): {1: 2}, (0, 2): {2: -2}, (1, 2): {0: 1}}),      # t0 = h, t1 = e, t2 = f
    "Heisenberg": structure({(1, 2): {0: 1}}),                                       # [t2, ty] = t1 with t1, t2, ty = 0, 1, 2
    "Bianchi V": structure({(2, 0): {0: 1}, (2, 1): {1: 1}}),
}
for name, f in algs.items():
    jac = all(sum(f(a, b, d) * f(d, c, e) + f(b, c, d) * f(d, a, e) + f(c, a, d) * f(d, b, e) for d in range(3)) == 0 for a in range(3) for b in range(3) for c in range(3) for e in range(3))
    print("%-11s Jacobi %s, tr ad(t_a) = %s, unimodular: %s" % (name, jac, [sum(f(a, b, b) for b in range(3)) for a in range(3)], all(sum(f(a, b, b) for b in range(3)) == 0 for a in range(3))))
