# name: cm-drag-limit-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly (test_fallback recipe, oracle
#   instances; the claim is a limit)
# goal: sage_test KaluzaKlein.ClassicalMechanics.Momentum.drag_limit file "cm-drag-limit-test.sage"
# description: lim (m v₀/γ)(1 - e^{-γ t/m}) = m v₀/γ as t → ∞, by Sage's limit at (m, γ, v₀) = (1, 1, 1) (edge), (2, 1/2, 3), (1, 3, -1)
SEED = 0  # the instances are listed by hand
t = SR.var('t')
def inst(m, g, v0, edge):
    return oracle_instance(str(m), str(g), str(v0), '_', '_',
                           lhs=limit(m * v0 / g * (1 - exp(-g * t / m)), t=oo), rhs=QQ(m * v0 / g), relation='==', edge=edge)
instances = [inst(1, 1, 1, True), inst(2, QQ(1) / 2, 3, False), inst(1, 3, -1, False)]
