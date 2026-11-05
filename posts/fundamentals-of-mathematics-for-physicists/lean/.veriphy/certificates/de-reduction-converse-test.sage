# name: de-reduction-converse-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.OrdinaryDifferentialEquations.first_order_to_reduction file "de-reduction-converse-test.sage"
# description: u = (cos, -sin) solves u₀' = u₁, u₁' = -u₀ (n = 1), u = (e^x, e^x, e^x) (n = 2, f = u₂): y^{(n+1)} = f at x = 0, 1
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('1', 'fun _ u => -u 0', '![Real.cos, fun x => -Real.sin x]', '_', '_', '0',
                    lhs=diff(cos(x), x, 2).subs(x=0),
                    rhs=-cos(0), relation='==', edge=True),
    oracle_instance('1', 'fun _ u => -u 0', '![Real.cos, fun x => -Real.sin x]', '_', '_', '1',
                    lhs=diff(cos(x), x, 2).subs(x=1),
                    rhs=-cos(1), relation='==', edge=False),
    oracle_instance('2', 'fun _ u => u 2', '![Real.exp, Real.exp, Real.exp]', '_', '_', '1',
                    lhs=diff(exp(x), x, 3).subs(x=1),
                    rhs=exp(1), relation='==', edge=False),
]
