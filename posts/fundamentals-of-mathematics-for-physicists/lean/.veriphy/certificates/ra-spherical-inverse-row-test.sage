# name: ra-spherical-inverse-row-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Calculus.spherical_inverse_row file "ra-spherical-inverse-row-test.sage"
# description: Sage multiplies the row (sin θ cos φ, sin θ sin φ, cos θ) by J and simplifies to (1, 0, 0)
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
r, th, ph = var('r th ph')
J = matrix(SR, [[sin(th) * cos(ph), r * cos(th) * cos(ph), -r * sin(th) * sin(ph)], [sin(th) * sin(ph), r * cos(th) * sin(ph), r * sin(th) * cos(ph)],
                [cos(th), -r * sin(th), 0]])
row = matrix(SR, [[sin(th) * cos(ph), sin(th) * sin(ph), cos(th)]])
P = (row * J).simplify_full()
instances = [O([str(a), str(b), str(c)], B(P.subs(r=a, th=b, ph=c) == matrix(SR, [[1, 0, 0]])), QQ(1), edge=bool(a == 1 and b == 0)) for a, b, c in [(1, 0, 0), (2, 1, 3), (-1, pi / 3, pi / 5)]]
