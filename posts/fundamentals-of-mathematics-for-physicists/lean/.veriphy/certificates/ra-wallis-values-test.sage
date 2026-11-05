# name: ra-wallis-values-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Exercises.wallis_values file "ra-wallis-values-test.sage"
# description: Sage's exact I_(2n) = (2n-1)!!/(2n)!! π/2 and I_(2n+1) = (2n)!!/(2n+1)!!
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
x = var('x')
dfact = lambda m: prod(range(m, 0, -2)) if m > 0 else 1
instances = []
for n in [0, 1, 2, 4]:
    ok = bool(integrate(sin(x) ** (2 * n), x, 0, pi / 2) == dfact(2 * n - 1) / dfact(2 * n) * pi / 2) and \
        bool(integrate(sin(x) ** (2 * n + 1), x, 0, pi / 2) == dfact(2 * n) / dfact(2 * n + 1))
    instances.append(O([str(n)], B(ok), QQ(1), edge=bool(n == 0)))
