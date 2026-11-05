# name: ra-gradient-orthogonal-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Calculus.gradient_orthogonal file "ra-gradient-orthogonal-test.sage"
# description: the ellipse point c(t) lies on f = 8 and ∇f(c(t)) · ċ(t) = 0, simplified exactly by Sage
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
t_ = var('t_')
instances = []
for tv in [0, pi / 4, 1, pi]:
    cx, cy = 2 * sqrt(2) * cos(t_), sqrt(2) * sin(t_)
    on = bool((cx ** 2 + 4 * cy ** 2 - 8).simplify_full() == 0)
    orth = bool((2 * cx * diff(cx, t_) + 8 * cy * diff(cy, t_)).simplify_full() == 0)
    instances.append(O([str(tv)], B(on and orth), QQ(1), edge=bool(tv == 0)))
