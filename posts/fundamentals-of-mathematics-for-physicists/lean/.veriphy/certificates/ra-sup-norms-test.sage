# name: ra-sup-norms-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.MultipleIntegrals.sup_norms file "ra-sup-norms-test.sage"
# description: Sage maximises |sin(nx)/n| (1/n at π/(2n)) and x/(1 + nx²) (1/(2√n) at 1/√n)
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
for n, xv in [(1, 1), (4, QQ(1) / 3), (9, -2), (100, QQ(1) / 10)]:
    h = x / (1 + n * x ** 2)
    crit = [s.rhs() for s in solve(diff(h, x) == 0, x)]
    mx = max(RI(h.subs(x=c)) for c in crit)
    ok = bool(sin(n * (pi / (2 * n))) / n == 1 / SR(n)) and abs(AA(QQ(h.subs(x=xv)))) <= 1 / (2 * AA(n).sqrt()) and \
        mx.overlaps(RI(1) / (2 * RI(n).sqrt()))
    instances.append(O([str(n), '(by norm_num)', str(xv)], B(ok), QQ(1), edge=bool(n == 1)))
