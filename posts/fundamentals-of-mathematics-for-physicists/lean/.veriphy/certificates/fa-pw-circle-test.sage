# name: fa-pw-circle-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.PeterWeyl.pw_circle file "fa-pw-circle-test.sage"
# description: Sage evaluates the orthonormality (1/T) ∫₀^T e^{2πinx/T} conj(e^{2πimx/T}) dx = δ_{nm} for T = 2π, 1
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
z_ = var('z_'); y = var('y'); assume(y, 'real')
instances = []
CASES = [(['2 * π', '⟨Real.two_pi_pos⟩'], True, {'T': '2*pi', 'n': '1', 'm': '1'}), (['2 * π', '_'], False, {'T': '2*pi', 'n': '2', 'm': '-1'}), (['1', '⟨one_pos⟩'], False, {'T': '1', 'n': '3', 'm': '3'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('1/T*integrate(exp(2*pi*I*n*z_/T)*exp(-2*pi*I*m*z_/T), z_, 0, T)', scope))
    R = side(eval('1 if bool(n == m) else 0', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
