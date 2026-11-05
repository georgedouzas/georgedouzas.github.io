# name: la-span-membership
# technique: rational witness (witness recipe), with the exercise's Sage solution as oracle
# goal: ∃ a b, (2, 5, 1) = a v₁ + b v₂ as component equations
# description: the oracle reruns the chapter's exercise script and prints its answer; the witness recipe then solves
#   the system read off the goal exactly over QQ, checks it against the oracle, and norm_num re-checks it in Lean.
# oracle (sage/exercises/exr-span-membership.sage)
v1, v2 = vector(QQ, [1, 2, 0]), vector(QQ, [0, 1, 1])
W = (QQ ^ 3).span([v1, v2])
assert vector(QQ, [2, 5, 1]) in W and vector(QQ, [1, 1, 1]) not in W
ORACLE = list(matrix(QQ, [v1, v2]).transpose().solve_right(vector(QQ, [2, 5, 1])))
print('oracle: (2, 5, 1) in span with coefficients', ORACLE, '; normal of the plane', matrix(QQ, [v1, v2]).right_kernel().basis()[0])
import itertools
gens = list(binders)
eqs = [c.poly for c in body if c.rel == 'eq']
les = [c.poly for c in body if c.rel == 'le']
lts = [c.poly for c in body if c.rel == 'lt']
nes = [c.poly for c in body if c.rel == 'ne']

def holds(pt):
    sub = dict(zip(gens, pt))
    return (all(q.subs(sub) == 0 for q in eqs) and all(q.subs(sub) <= 0 for q in les)
            and all(q.subs(sub) < 0 for q in lts) and all(q.subs(sub) != 0 for q in nes))

def linear_point():
    # x = x0 + t K over the solutions of the equations; an exact LP places t inside the inequalities
    n = len(gens)
    A = matrix(QQ, [[q.monomial_coefficient(x) for x in gens] for q in eqs]) if eqs else matrix(QQ, 0, n)
    b = vector(QQ, [-q.constant_coefficient() for q in eqs]) if eqs else vector(QQ, [])
    x0 = A.solve_right(b) if eqs else vector(QQ, [0] * n)
    K = A.right_kernel().basis_matrix() if eqs else matrix.identity(QQ, n)
    k = K.nrows()
    def in_t(q):
        return ([sum(q.monomial_coefficient(x) * K[r, j] for j, x in enumerate(gens)) for r in range(k)],
                q.constant_coefficient() + sum(q.monomial_coefficient(x) * x0[j] for j, x in enumerate(gens)))
    base = [QQ(0)] * k
    if les or lts:
        rows = [(cs + [0], c0) for cs, c0 in map(in_t, les)] + [(cs + [1], c0) for cs, c0 in map(in_t, lts)]
        sol = solve_exact_lp(k + 1, les=rows + [([0] * k + [1], -1)], objective=[0] * k + [1], nonneg=[k], maximize=True)
        assert sol is not None, 'the inequalities are infeasible'
        base = sol[:k]
    span = min(k, 6)
    for scale in (1, 2, 3, 5, 7, 11):
        for d in itertools.product(range(-1, 2), repeat=span):
            c = [base[r] + QQ((list(d) + [0] * k)[r]) / scale for r in range(k)]
            pt = [x0[j] + sum(K[r, j] * c[r] for r in range(k)) for j in range(n)]
            if holds(pt):
                return pt
    return None

def nonlinear_point():
    if len(gens) == 1 and len(eqs) == 1:
        for r, _ in eqs[0].univariate_polynomial().roots(QQ):
            if holds([r]):
                return [r]
    sv = [SR.var(str(g)) for g in gens]
    for sol in solve([SR(str(q)) == 0 for q in eqs], sv, solution_dict=True):
        try:
            pt = [QQ(sol[v]) for v in sv]
        except (TypeError, ValueError, KeyError):
            continue
        if holds(pt):
            return pt
    return None

linear = all(q.degree() <= 1 for q in eqs + les + lts)
point = linear_point() if linear else nonlinear_point()
assert point is not None, 'no rational witness found'
proof = witness(point, norm_num())
assert ORACLE is None or [QQ(c) for c in point] == [QQ(c) for c in ORACLE], f'witness {point} differs from the oracle {ORACLE}'
