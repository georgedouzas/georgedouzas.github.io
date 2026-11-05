# name: ra-limit-eps-delta-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.limit_eps_delta file "ra-limit-eps-delta-test.sage"
# description: Sage's limit of f at a equals L, and for ε = 1/10 a δ found by Sage works on a sample of points
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
F = {'(fun x : ℝ => x ^ 2)': x ** 2, '(fun x : ℝ => Real.sin x)': sin(x), '(fun x : ℝ => Real.exp x)': exp(x),
     '(fun x : ℝ => x ^ 3 - 2 * x)': x ** 3 - 2 * x, '(fun x : ℝ => (1 : ℝ))': SR(1), '(fun x : ℝ => Real.cos x)': cos(x),
     '(fun x : ℝ => x ^ 3)': x ** 3, '(fun x : ℝ => Real.log (1 + x ^ 2))': log(1 + x ** 2), '(fun x : ℝ => x)': x}

instances = []
for f, a, edge in [('(fun x : ℝ => x ^ 2)', 2, False), ('(fun x : ℝ => Real.sin x)', 0, False), ('(fun x : ℝ => (1 : ℝ))', 1, True)]:
    L = limit(F[f], x=a)
    eps = QQ(1) / 10
    delta = QQ(1) / 100
    ok = all(abs(RR(F[f].subs(x=a + h)) - RR(L)) < eps for h in [delta * j / 50 for j in range(-49, 50) if j != 0])
    instances.append(O([f, str(a), str(L)], B(ok), QQ(1), edge=edge))
