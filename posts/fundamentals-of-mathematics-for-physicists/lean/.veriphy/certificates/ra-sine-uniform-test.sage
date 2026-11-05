# name: ra-sine-uniform-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.MultipleIntegrals.sine_uniform file "ra-sine-uniform-test.sage"
# description: |sin(nx)/n| ≤ 1/n by intervals and Sage's exact ∫₀^π sin(nx)/n dx = (1 - cos nπ)/n²
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
instances = []
for n, xv, edge in [(1, 0, True), (2, 1, False), (3, -2, False)]:
    ok = abs(R(sin(n * xv))) / n <= R(1) / n and bool(integrate(sin(n * x) / n, x, 0, pi) == (1 - cos(n * pi)) / n ** 2)
    instances.append(O([str(n), '(by norm_num)', str(xv)], B(ok), QQ(1), edge=edge))
