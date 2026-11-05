# name: de-poisson-kernel-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.SeparationOfVariables.poisson_kernel file "de-poisson-kernel-test.sage"
# description: Σ 2rⁿ⁺¹cos((n+1)α) = (1 - r²)/(1 - 2r cos α + r²) - 1 by Sage's series sum at (r, α) = (0, 1) (edge), (1/2, 0), (1/3, π/2)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
n_ = var('n_')
instances = [
    oracle_instance('0', '1', '_', '_',
                    lhs=SR(sum(2*(0)^(n_+1)*cos((n_+1)*(1)), n_, 0, oo)).simplify_full(),
                    rhs=SR((1 - (0)^2)/(1 - 2*(0)*cos(1) + (0)^2) - 1).simplify_full(), relation='==', edge=True),
    oracle_instance('1/2', '0', '_', '_',
                    lhs=SR(sum(2*(1/2)^(n_+1)*cos((n_+1)*(0)), n_, 0, oo)).simplify_full(),
                    rhs=SR((1 - (1/2)^2)/(1 - 2*(1/2)*cos(0) + (1/2)^2) - 1).simplify_full(), relation='==', edge=False),
    oracle_instance('1/3', 'pi/2', '_', '_',
                    lhs=SR(sum(2*(1/3)^(n_+1)*cos((n_+1)*(pi/2)), n_, 0, oo)).simplify_full(),
                    rhs=SR((1 - (1/3)^2)/(1 - 2*(1/3)*cos(pi/2) + (1/3)^2) - 1).simplify_full(), relation='==', edge=False),
]
