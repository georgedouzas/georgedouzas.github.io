# name: fa-law-of-cosines-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.ChapterExercises.law_of_cosines file "fa-law-of-cosines-test.sage"
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
CASES = [(['0', '0', 'e₃', 'e₃', '_', '_'], True, {'r': '0', 's': '0', 'c': '1'}), (['1', '2', 'e₃', 'rotated', '_', '_'], False, {'r': '1', 's': '2', 'c': '1/2'}), (['3', '1/2', 'e₁', 'e₂', '_', '_'], False, {'r': '3', 's': '1/2', 'c': '0'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('(r*vector([0, 0, 1]) - s*vector([sqrt(1 - c**2), 0, c])).dot_product(r*vector([0, 0, 1]) - s*vector([sqrt(1 - c**2), 0, c]))', scope))
    R = side(eval('r**2 + s**2 - 2*r*s*c', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
