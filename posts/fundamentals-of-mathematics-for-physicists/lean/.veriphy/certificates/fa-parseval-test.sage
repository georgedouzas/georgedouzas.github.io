# name: fa-parseval-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.Parseval.parseval file "fa-parseval-test.sage"
# description: Sage computes the coefficients symbolically (c_0 apart) and sums Σ|c_n|² for f = 1, x, x²
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
assume(n, 'integer')
def csum(f, a, b):
    L = b - a
    c = lambda m: 1/L*integrate(f(y)*exp(-2*pi*I*m*y/L), y, a, b)
    cn = (c(n)*conjugate(c(n))).simplify_full()
    return abs(c(0))**2 + sum(cn, n, 1, oo) + sum(cn.subs(n=-n), n, 1, oo)
instances = []
CASES = [(['0', '1', 'zero_lt_one', 'fun _ => 1', '_'], True, {'f': 'lambda t: 1', 'a': '0', 'b': '1'}), (['0', '2 * π', '_', 'fun x => (x ^ 2 : ℂ)', '_'], False, {'f': 'lambda t: t**2', 'a': '0', 'b': '2*pi'}), (['0', '2 * π', '_', 'fun x => (x : ℂ)', '_'], False, {'f': 'lambda t: t', 'a': '0', 'b': '2*pi'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('csum(f, a, b)', scope))
    R = side(eval('1/(b - a)*integrate(abs(f(y))**2, y, a, b)', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
