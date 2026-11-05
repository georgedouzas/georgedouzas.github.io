# name: ra-gaussian-shifted-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.MultipleIntegrals.gaussian_shifted file "ra-gaussian-shifted-test.sage"
# description: Sage's ∫ e^(-(x-μ)²/2σ²) dx = √(2π) σ
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
instances = [O([str(mu), str(sg), '(by norm_num)'], integrate(exp(-(x - mu) ** 2 / (2 * sg ** 2)), x, -oo, oo), sqrt(2 * pi) * sg, edge=bool(mu == 0 and sg == 1))
             for mu, sg in [(0, 1), (3, 2), (-1, QQ(1) / 2)]]
