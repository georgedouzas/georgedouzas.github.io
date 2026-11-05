# name: fa-positive-kernel-bound-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.PartialSums.positive_kernel_bound file "fa-positive-kernel-bound-test.sage"
# description: Sage evaluates |(1/2π) ∫ F(x - y) f(y) dy| ≤ M for non-negative kernels of mass 2π; the edge case f = M attains it
SEED = 0
RI = RealIntervalField(128)
CI = ComplexIntervalField(128)
def num(e):
    try:
        return RI(e)
    except (TypeError, ValueError):
        pass
    e = SR(e)
    for f in (lambda t: t, lambda t: t.simplify_full(), lambda t: t.canonicalize_radical()):
        try:
            return RI(f(e))
        except (TypeError, ValueError):
            pass
    try:
        return CI(e)
    except (TypeError, ValueError):
        return RI(e.n(prec=256))
def side(e):
    v = num(e)
    return v
y = var('y')
instances = []
CASES = [(['fun _ => 1', 'fun _ => 1', '_', '_', '_', '1', '_', '0', '_'], True, {'F': 'lambda z: 1', 'f': 'lambda z: 1', 'M': '1', 'x': '0'}), (['fun t => 1 + Real.cos t', 'Real.sin', '_', '_', '_', '1', '_', '1', '_'], False, {'F': 'lambda z: 1 + cos(z)', 'f': 'lambda z: sin(z)', 'M': '1', 'x': '1'}), (['fun t => 1 - Real.cos t', 'fun y => 2 * Real.cos y', '_', '_', '_', '2', '_', '1/2', '_'], False, {'F': 'lambda z: 1 - cos(z)', 'f': 'lambda z: 2*cos(z)', 'M': '2', 'x': '1/2'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('abs(1/(2*pi)*integrate(F(x - y)*f(y), y, -pi, pi))', scope))
    R = side(eval('M', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='<=', edge=edge))
    print(values, L, R)
