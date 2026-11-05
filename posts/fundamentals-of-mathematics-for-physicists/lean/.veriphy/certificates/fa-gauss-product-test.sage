# name: fa-gauss-product-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.ConvolutionProduct.gauss_product file "fa-gauss-product-test.sage"
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
instances = []
CASES = [(['1', '1', '0', 'one_pos', 'one_pos'], True, {'s': '1', 't': '1', 'k': '0'}), (['2', '3', '1', '_', '_'], False, {'s': '2', 't': '3', 'k': '1'}), (['1/2', '1', '-2', '_', '_'], False, {'s': '1/2', 't': '1', 'k': '-2'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('(s*sqrt(2*pi)*exp(-s**2*k**2/2))*(t*sqrt(2*pi)*exp(-t**2*k**2/2))', scope))
    R = side(eval('(s*t*sqrt(2*pi)/sqrt(s**2 + t**2))*(sqrt(s**2 + t**2)*sqrt(2*pi)*exp(-(s**2 + t**2)*k**2/2))', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
