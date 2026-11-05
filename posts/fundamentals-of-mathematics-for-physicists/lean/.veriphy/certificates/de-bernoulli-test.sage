# name: de-bernoulli-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.LinearFirstOrder.bernoulli_substitution file "de-bernoulli-test.sage"
# description: u = y^{1-n} solves u' + (1 - n)pu = (1 - n)q for y = 1/(1 + eˣ) (n = 2, p = q = 1), y = 1 (edge), at x = 0, 1
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('2', 'fun _ => 1', 'fun _ => 1', 'fun x => 1 / (1 + Real.exp x)', '0', '_', '_', '_', '_',
                    lhs=diff((1/(1+exp(x)))^(-1), x).subs(x=0),
                    rhs=(-1)*1 - (-1)*1*(1/(1+exp(0)))^(-1), relation='==', edge=False),
    oracle_instance('2', 'fun _ => 1', 'fun _ => 1', 'fun x => 1 / (1 + Real.exp x)', '1', '_', '_', '_', '_',
                    lhs=diff((1/(1+exp(x)))^(-1), x).subs(x=1),
                    rhs=(-1)*1 - (-1)*1*(1/(1+exp(1)))^(-1), relation='==', edge=False),
    oracle_instance('3', 'fun _ => 1', 'fun _ => 1', 'fun _ => 1', '0', '0', '_', '_', '_',
                    lhs=diff(SR(1)^(-2), x),
                    rhs=(-2)*1 - (-2)*1*1, relation='==', edge=True),
]
