# name: ra-change-of-variables-polar-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.MultipleIntegrals.change_of_variables_polar file "ra-change-of-variables-polar-test.sage"
# description: Sage evaluates both sides for radial f: ∫∫ f dx dy = ∫∫ r f(r cos φ, r sin φ) dr dφ (π, π/2, π, 0)
SEED = 0  # the instances are listed, no sampling
R = RealIntervalField(128)
def B(b):
    return QQ(1 if bool(b) else 0)
RI = RealIntervalField(256)
def O(values, lhs, rhs, rel='==', edge=False):
    # symbolic values (π, e, logs, limits Sage computed) are compared by 256-bit intervals; rationals exactly
    if any(hasattr(v, 'parent') and v.parent() is SR for v in (lhs, rhs)):
        lhs, rhs = RI(lhs), RI(rhs)
    return oracle_instance(*values, lhs=lhs, rhs=rhs, relation=rel, edge=edge)
x, y, r, ph = var('x y r ph')
instances = []
for f, fpol, edge in [(exp(-x ** 2 - y ** 2), exp(-r ** 2), False), (exp(-2 * x ** 2 - 2 * y ** 2), exp(-2 * r ** 2), False),
                      ((x ** 2 + y ** 2) * exp(-x ** 2 - y ** 2), r ** 2 * exp(-r ** 2), False), (SR(0), SR(0), True)]:
    lhs = integrate(integrate(f, x, -oo, oo), y, -oo, oo)
    rhs = integrate(integrate(r * fpol, r, 0, oo), ph, -pi, pi)
    instances.append(O([f'(fun p : ℝ × ℝ => {f})'], lhs, rhs, edge=edge))
