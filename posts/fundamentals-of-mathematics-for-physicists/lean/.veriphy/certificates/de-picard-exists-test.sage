# name: de-picard-exists-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.ExistenceUniqueness.picard_exists file "de-picard-exists-test.sage"
# description: the Picard-Lindelöf solution exists: y' = y on [0, 1/2] around (0, 1) (y = eᵗ), y' = 0 (edge, constant), y' = t on [-1, 1] (y = t²/2)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('ℝ', '_', '_', '_', 'fun _ y => y', '0', '1/2', '⟨0, _⟩', '1', '1', '2', '1', '_',
                    lhs=B(diff(exp(x), x) - exp(x) == 0 and exp(0) == 1),
                    rhs=1, relation='==', edge=False),
    oracle_instance('ℝ', '_', '_', '_', 'fun _ _ => 0', '0', '1', '⟨0, _⟩', '0', '1', '0', '0', '_',
                    lhs=B(diff(SR(0), x) == 0),
                    rhs=1, relation='==', edge=True),
    oracle_instance('ℝ', '_', '_', '_', 'fun t _ => t', '-1', '1', '⟨0, _⟩', '0', '1', '1', '0', '_',
                    lhs=B(diff(x^2/2, x) - x == 0),
                    rhs=1, relation='==', edge=False),
]
