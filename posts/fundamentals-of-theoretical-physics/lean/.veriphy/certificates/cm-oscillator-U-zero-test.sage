# name: cm-oscillator-U-zero-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly (test_fallback recipe, oracle
#   instances; the claim is a one-sided limit)
# goal: sage_test KaluzaKlein.ClassicalMechanics.CentralForces.oscillator_U_tendsto_zero file "cm-oscillator-U-zero-test.sage"
# description: U = L²/(2mr²) + kr²/2 → ∞ as r → 0⁺, by Sage's limit at (m, k, L) = (1, 0, 1) (edge, k = 0), (2, 1/2, 3), (1, 3, -1)
SEED = 0  # the instances are listed by hand
r = SR.var('r')
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
def inst(m, k, L, edge):
    return oracle_instance(str(m), str(k), str(L), '_', '_', '_',
                           lhs=B(limit(L**2 / (2 * m * r**2) + k * r**2 / 2, r=0, dir='+') == oo), rhs=1, relation='==', edge=edge)
instances = [inst(1, 0, 1, True), inst(2, QQ(1) / 2, 3, False), inst(1, 3, -1, False)]
