# name: ra-gaussian-second-moment-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.MultipleIntegrals.gaussian_second_moment file "ra-gaussian-second-moment-test.sage"
# description: Sage's -d/da √(π/a) and ∫ x² e^(-ax²) dx = ½√π a^(-3/2)
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
x, a = var('x a')
instances = []
for av in [1, 2, QQ(1) / 2]:
    d = diff(sqrt(pi / a), a).subs(a=av); m = integrate(x ** 2 * exp(-av * x ** 2), x, -oo, oo)
    ok = bool((d + sqrt(pi) / 2 * SR(av) ** (-QQ(3) / 2)).simplify_full() == 0) and bool((m - sqrt(pi) / 2 * SR(av) ** (-QQ(3) / 2)).simplify_full() == 0)
    instances.append(O([str(av), '(by norm_num)'], B(ok), QQ(1), edge=bool(av == 1)))
