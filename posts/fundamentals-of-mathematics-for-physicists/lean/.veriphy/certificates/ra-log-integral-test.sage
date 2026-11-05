# name: ra-log-integral-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Integral.log_integral file "ra-log-integral-test.sage"
# description: Sage's ∫₁ˣ dt/t = log x
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
t_ = var('t_')
instances = [O([str(v), '(by norm_num)'], integrate(1 / t_, t_, 1, v), log(SR(v)), edge=bool(v == 1)) for v in [1, 2, QQ(1) / 3, 7]]
