# name: fa-sine-dirichlet-mode-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.SeriesComputations.sine_dirichlet_mode file "fa-sine-dirichlet-mode-test.sage"
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
y = var('y'); k = var('k'); i = var('i'); n = var('n'); z = var('z'); v = var('v')
instances = []
CASES = [(['0', '1'], True, {'n': '0', 'x': '1'}), (['2', '1/3'], False, {'n': '2', 'x': '1/3'}), (['5', '-2'], False, {'n': '5', 'x': '-2'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('abs(diff(sin(n*z), z, 2).subs(z=x) + n**2*sin(n*x)) + abs(sin(n*0)) + abs(sin(n*pi))', scope))
    R = side(eval('0', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
