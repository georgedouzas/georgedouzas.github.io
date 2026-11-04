# Exercise exr-ren-hopf-rota-baxter: the Rota-Baxter identity R(fg) + R(f)R(g) = R(R(f)g) + R(fR(g)) for the pole part K
# on Laurent polynomials in eps (stored as dictionaries exponent -> coefficient), for the evaluation at s0, and its
# failure for the projection on the eps^0 term.
def mul(f, g):
    h = {}
    for i, a in f.items():
        for j, b in g.items():
            h[i + j] = h.get(i + j, 0) + a * b
    return {k: v for k, v in h.items() if v != 0}
def add(*fs):
    h = {}
    for f in fs:
        for k, v in f.items(): h[k] = h.get(k, 0) + v
    return {k: v for k, v in h.items() if v != 0}
neg = lambda f: {k: -v for k, v in f.items()}
K = lambda f: {k: v for k, v in f.items() if k < 0}
P0 = lambda f: {0: f[0]} if f.get(0, 0) != 0 else {}
rb = lambda op, f, g: add(op(mul(f, g)), mul(op(f), op(g)), neg(op(mul(op(f), g))), neg(op(mul(f, op(g)))))
set_random_seed(3)
fs = [{k: QQ.random_element(num_bound=5, den_bound=3) for k in range(-3, 3)} for _ in range(4)]
pairs = [(f, g) for f in fs for g in fs if f is not g]
print("pole part K: Rota-Baxter defects vanish on all %d pairs:" % len(pairs), all(rb(K, f, g) == {} for f, g in pairs))
print("eps^0 projection: defects on the first three pairs:", [rb(P0, f, g) for f, g in pairs[:3]])
var('s'); f1, g1 = 1 + s + s^2, 2 - s^3
Ev = lambda h: SR(h).subs(s=1)
print("evaluation at s0 = 1: defect", Ev(f1 * g1) + Ev(f1) * Ev(g1) - Ev(Ev(f1) * g1) - Ev(f1 * Ev(g1)))
