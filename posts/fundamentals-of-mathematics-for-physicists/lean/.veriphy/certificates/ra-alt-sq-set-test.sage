# name: ra-alt-sq-set-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Series.alt_sq_set file "ra-alt-sq-set-test.sage"
# description: ∑ (-1)^k x^(2k) converges exactly on (-1, 1): Sage sums it inside, the terms do not vanish outside
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
k = var('k')
instances = []
for x, edge in [(0, False), (QQ(1) / 2, False), (1, True), (-2, False)]:
    conv = abs(x) < 1 and (x == 0 or bool(sum((-1) ** k * SR(x) ** (2 * k), k, 0, oo) == 1 / (1 + x ** 2)))
    instances.append(O([str(x)], B(conv), B(abs(x) < 1), edge=edge))
