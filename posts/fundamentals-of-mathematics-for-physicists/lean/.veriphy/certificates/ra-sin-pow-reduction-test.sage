# name: ra-sin-pow-reduction-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Integral.sin_pow_reduction file "ra-sin-pow-reduction-test.sage"
# description: Sage's I_n = ∫₀^(π/2) sinⁿ satisfies I_(n+2) = (n+1)/(n+2) I_n
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
instances = [O([str(n)], integrate(sin(x) ** (n + 2), x, 0, pi / 2), (n + 1) / SR(n + 2) * integrate(sin(x) ** n, x, 0, pi / 2), edge=bool(n == 0))
             for n in [0, 1, 2, 5]]
