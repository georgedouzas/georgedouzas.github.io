# name: ra-deriv-slope-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.deriv_slope file "ra-deriv-slope-test.sage"
# description: Sage's limit of the difference quotient equals Sage's derivative
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

h = var('h')
instances = []
for f, a, edge in [('(fun x : ℝ => x ^ 2)', 3, False), ('(fun x : ℝ => Real.exp x)', 0, False), ('(fun x : ℝ => (1 : ℝ))', 2, True),
                   ('(fun x : ℝ => Real.sin x)', 1, False)]:
    d = diff(F[f], x).subs(x=a)
    q = limit((F[f].subs(x=a + h) - F[f].subs(x=a)) / h, h=0)
    instances.append(O([f, str(a), str(d)], q, d, edge=edge))
