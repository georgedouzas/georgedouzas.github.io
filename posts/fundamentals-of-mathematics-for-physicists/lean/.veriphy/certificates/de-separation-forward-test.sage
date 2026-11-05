# name: de-separation-forward-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.SeparableEquations.separation_forward file "de-separation-forward-test.sage"
# description: y' = f g(y) recovered from H(y) = F + C: y = tan x (H = arctan, g = 1 + y², F = x, f = 1) and y = Ce^x (H = log, g = y), at x = 0 (edge), 1
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('Real.arctan', 'id', 'Real.tan', 'fun y => 1 + y ^ 2', 'fun _ => 1', '1', '0', '0', '_', '_', '_', '_', '_',
                    lhs=diff(tan(x), x).subs(x=0),
                    rhs=1*(1+tan(0)^2), relation='==', edge=True),
    oracle_instance('Real.arctan', 'id', 'Real.tan', 'fun y => 1 + y ^ 2', 'fun _ => 1', '_', '0', '1', '_', '_', '_', '_', '_',
                    lhs=diff(tan(x), x).subs(x=1),
                    rhs=1+tan(1)^2, relation='==', edge=False),
    oracle_instance('Real.log', 'id', 'fun x => 2 * Real.exp x', 'id', 'fun _ => 1', '_', 'Real.log 2', '1', '_', '_', '_', '_', '_',
                    lhs=diff(2*exp(x), x).subs(x=1),
                    rhs=1*2*exp(1), relation='==', edge=False),
]
