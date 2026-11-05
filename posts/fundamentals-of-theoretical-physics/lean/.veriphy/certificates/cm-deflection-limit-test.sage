# name: cm-deflection-limit-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly (test_fallback recipe, oracle
#   instances; the claim is a limit)
# goal: sage_test KaluzaKlein.ClassicalMechanics.Kepler.deflection_limit file "cm-deflection-limit-test.sage"
# description: 2 arccos (-c/e) - π → 0 as e → ∞, by Sage's limit at c = 0 (edge), 1, 5
SEED = 0  # the instances are listed by hand
e = SR.var('e')
def inst(c, edge):
    return oracle_instance(str(c), lhs=limit(2 * arccos(-c / e) - pi, e=oo), rhs=0, relation='==', edge=edge)
instances = [inst(0, True), inst(1, False), inst(5, False)]
