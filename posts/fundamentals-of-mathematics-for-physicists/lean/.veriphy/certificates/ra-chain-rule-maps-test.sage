# name: ra-chain-rule-maps-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Calculus.chain_rule_maps file "ra-chain-rule-maps-test.sage"
# description: Sage's Jacobian of G ∘ F equals J_G(F(p)) J_F(p) for the polar map, a rotation and the squaring map
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
r, ph, x, y, al = var('r ph x y al')
instances = []
for F, G, p, edge in [((r * cos(ph), r * sin(ph)), (x * cos(1) - y * sin(1), x * sin(1) + y * cos(1)), (2, QQ(1) / 3), False),
                      ((r ** 2 - ph ** 2, 2 * r * ph), (x + y, x * y), (1, 2), False), ((r, ph), (x, y), (0, 0), True)]:
    JF = jacobian(list(F), [r, ph]); JG = jacobian(list(G), [x, y])
    comp = [g.subs(x=F[0], y=F[1]) for g in G]
    lhs = jacobian(comp, [r, ph]).subs(r=p[0], ph=p[1])
    rhs = JG.subs(x=F[0], y=F[1]).subs(r=p[0], ph=p[1]) * JF.subs(r=p[0], ph=p[1])
    ok = all(bool((lhs[i, j] - rhs[i, j]).simplify_full() == 0) for i in range(2) for j in range(2))
    instances.append(O([str(F), str(G), str(p), '‹_›', '‹_›', '‹_›', '‹_›'], B(ok), QQ(1), edge=edge))
