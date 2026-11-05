# name: cm-hyperbola-sweep-test
# technique: test a claim no recipe certifies, at instances Sage evaluates in interval arithmetic (test_fallback recipe)
# goal: sage_test KaluzaKlein.ClassicalMechanics.Kepler.hyperbola_sweep file "cm-hyperbola-sweep-test.sage"
# description: π < 2 arccos (-1/e) for e > 1, at e = 101/100 (edge, nearly parabolic), 2, 10
SEED = 0  # the instances are listed by hand
RIF = RealIntervalField(64)
def inst(e, edge):
    return oracle_instance(str(e), '_', lhs=RIF(pi), rhs=RIF(2 * arccos(-1 / QQ(e))), relation='<', edge=edge)
instances = [inst(QQ(101) / 100, True), inst(2, False), inst(10, False)]
