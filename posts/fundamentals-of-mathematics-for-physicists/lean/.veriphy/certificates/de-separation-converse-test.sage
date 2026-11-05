# name: de-separation-converse-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.SeparableEquations.separation_converse file "de-separation-converse-test.sage"
# description: H(y) - F has zero derivative on solutions: (H, F, y, g, f) = (arctan, x, tan, 1 + y², 1), (log, x, 2eˣ, y, 1), at x = 0 (edge), 1/2
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('Real.arctan', 'id', 'Real.tan', 'fun y => 1 + y ^ 2', 'fun _ => 1', '0', '_', '_', '_', '_',
                    lhs=diff(arctan(tan(x)) - x, x).simplify_full().subs(x=0),
                    rhs=0, relation='==', edge=True),
    oracle_instance('Real.arctan', 'id', 'Real.tan', 'fun y => 1 + y ^ 2', 'fun _ => 1', '1/2', '_', '_', '_', '_',
                    lhs=diff(arctan(tan(x)) - x, x).simplify_full().subs(x=1/2),
                    rhs=0, relation='==', edge=False),
    oracle_instance('Real.log', 'id', 'fun x => 2 * Real.exp x', 'id', 'fun _ => 1', '1/2', '_', '_', '_', '_',
                    lhs=diff(log(2*exp(x)) - x, x).simplify_full().subs(x=1/2),
                    rhs=0, relation='==', edge=False),
]
