# name: de-bernoulli-example-solves-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.LinearFirstOrder.bernoulli_example_solves file "de-bernoulli-example-solves-test.sage"
# description: y = 1/(1 + eˣ) has y' + y = y² by Sage at x = 0 (edge), 1, -2
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
s_ = var('s_')
instances = [
    oracle_instance('fun s => 1 / (1 + Real.exp s)', '_', '0',
                    lhs=(diff(1/(1+exp(s_)), s_) - ((1/(1+exp(s_)))^2 - 1/(1+exp(s_)))).subs(s_=0).simplify_full(),
                    rhs=0, relation='==', edge=True),
    oracle_instance('fun s => 1 / (1 + Real.exp s)', '_', '1',
                    lhs=(diff(1/(1+exp(s_)), s_) - ((1/(1+exp(s_)))^2 - 1/(1+exp(s_)))).subs(s_=1).simplify_full(),
                    rhs=0, relation='==', edge=False),
    oracle_instance('fun s => 1 / (1 + Real.exp s)', '_', '-2',
                    lhs=(diff(1/(1+exp(s_)), s_) - ((1/(1+exp(s_)))^2 - 1/(1+exp(s_)))).subs(s_=-2).simplify_full(),
                    rhs=0, relation='==', edge=False),
]
