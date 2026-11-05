# name: fa-squarewave-of-pos-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.PeriodicFunctions.squareWave_of_pos file "fa-squarewave-of-pos-test.sage"
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

def toioc(x):
    x = SR(x)
    k = ceil(RI(((x - pi) / (2 * pi))).upper()) if False else None
    k = floor(RI((pi - x) / (2 * pi)).lower())
    y = x + 2 * pi * k
    while bool(RI(y).upper() <= RI(-pi).lower()):
        y += 2 * pi
    while bool(RI(y).lower() > RI(pi).upper()):
        y -= 2 * pi
    return y
def sq(x):
    return sign(toioc(x))
def saw(x):
    return toioc(x)
def absw(x):
    return abs(toioc(x))

instances = []
CASES = [(['1/100', '_', '_'], True, {'x': '1/100'}), (['1', '_', '_'], False, {'x': '1'}), (['3', '_', '_'], False, {'x': '3'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('sq(x)', scope))
    R = side(eval('1', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
