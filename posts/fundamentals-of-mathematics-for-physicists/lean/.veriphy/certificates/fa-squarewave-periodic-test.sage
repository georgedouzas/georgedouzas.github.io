# name: fa-squarewave-periodic-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.PeriodicFunctions.squareWave_periodic file "fa-squarewave-periodic-test.sage"
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
CASES = [(['0'], True, {'x': '0'}), (['1'], False, {'x': '1'}), (['-5/2'], False, {'x': '-5/2'}), (['7'], False, {'x': '7'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('sq(x + 2*pi)', scope))
    R = side(eval('sq(x)', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
