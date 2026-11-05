# name: ra-product-rule-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.product_rule file "ra-product-rule-test.sage"
# description: Sage differentiates the product rule side and compares with the formula
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
pairs = [('(fun x : ℝ => x ^ 2)', '(fun x : ℝ => Real.sin x)', 1, False), ('(fun x : ℝ => Real.exp x)', '(fun x : ℝ => x ^ 3 - 2 * x)', 2, False),
         ('(fun x : ℝ => (1 : ℝ))', '(fun x : ℝ => Real.cos x)', 0, True)]
for f, g, a, edge in pairs:
    fe, ge = F[f], F[g]
    op = '*'
    if op == '*':
        lhs = diff(fe * ge, x).subs(x=a); rhs = (diff(fe, x) * ge + fe * diff(ge, x)).subs(x=a)
        vals = [f, g, str(a), str(diff(fe, x).subs(x=a)), str(diff(ge, x).subs(x=a)), '‹_›', '‹_›']
    elif op == '/':
        lhs = diff(fe / ge, x).subs(x=a); rhs = ((diff(fe, x) * ge - fe * diff(ge, x)) / ge ** 2).subs(x=a)
        vals = [f, g, str(a), str(diff(fe, x).subs(x=a)), str(diff(ge, x).subs(x=a)), '‹_›', '‹_›', '(by norm_num)']
    else:
        lhs = diff(ge.subs(x=fe), x).subs(x=a); rhs = (diff(ge, x).subs(x=fe) * diff(fe, x)).subs(x=a)
        vals = [f, g, str(a), str(diff(fe, x).subs(x=a)), str(diff(ge, x).subs(x=fe.subs(x=a))), '‹_›', '‹_›']
    instances.append(O(vals, lhs, rhs, edge=edge))
