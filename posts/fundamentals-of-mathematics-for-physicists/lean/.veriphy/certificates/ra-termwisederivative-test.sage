# name: ra-termwisederivative-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Series.TermwiseDerivative file "ra-termwisederivative-test.sage"
# description: Sage differentiates the sums of ∑ x^k (R = 1) and ∑ x^k/k! (R = ∞) and the term-by-term series, at sample points
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
x, k = var('x k')
instances = []
for c, R0, pt, edge in [('(fun _ : ℕ => (1 : ℝ))', '1', QQ(1) / 2, False), ('(fun k : ℕ => 1 / (k ! : ℝ))', '10', 1, False),
                        ('(fun _ : ℕ => (0 : ℝ))', '1', 0, True)]:
    if c.startswith('(fun _ : ℕ => (1'):
        f = 1 / (1 - x); termwise = sum((k + 1) * x ** k, k, 0, oo)
    elif 'k !' in c:
        f = exp(x); termwise = exp(x)   # ∑ (k+1) x^k/(k+1)! = ∑ x^k/k!
    else:
        f = 0 * x; termwise = 0 * x
    instances.append(O([c, R0, '‹_›', str(pt), '(by norm_num)'], diff(f, x).subs(x=pt), termwise.subs(x=pt), edge=edge))
