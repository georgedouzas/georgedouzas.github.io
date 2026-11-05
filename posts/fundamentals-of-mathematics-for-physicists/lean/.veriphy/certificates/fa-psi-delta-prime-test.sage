# name: fa-psi-delta-prime-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.DistributionDerivatives.psi_delta_prime file "fa-psi-delta-prime-test.sage"
# description: no recipe certifies the general statement; instances with an edge case first
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
z_ = var('z_')
instances = []
CASES = [(['fun x => x', 'Real.cos', '_', '_'], True, {'s': 'lambda t: t', 'p': 'lambda t: cos(t)'}), (['Real.exp', 'Real.sin', '_', '_'], False, {'s': 'lambda t: exp(t)', 'p': 'lambda t: sin(t)'}), (['fun x => x ^ 2 + 1', 'fun x => x + 2', '_', '_'], False, {'s': 'lambda t: t**2 + 1', 'p': 'lambda t: t + 2'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('-diff(s(z_)*p(z_), z_).subs(z_=0)', scope))
    R = side(eval('s(0)*(-diff(p(z_), z_).subs(z_=0)) - diff(s(z_), z_).subs(z_=0)*p(0)', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
