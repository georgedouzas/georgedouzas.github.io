# name: de-string-increasing-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.SturmLiouvilleTheorem.string_eigenvalues_increasing file "de-string-increasing-test.sage"
# description: the eigenvalues ((n+1)π/L)² increase: consecutive values compared exactly for L = π (edge: n²), 1, 3
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
n_ = var('n_')
instances = [
    oracle_instance('Real.pi', '_',
                    lhs=B(all(((n+1)*pi/pi)^2 < ((n+2)*pi/pi)^2 for n in range(10))),
                    rhs=1, relation='==', edge=True),
    oracle_instance('1', '_',
                    lhs=B(all(bool(((n+1)*pi)^2 < ((n+2)*pi)^2) for n in range(10))),
                    rhs=1, relation='==', edge=False),
    oracle_instance('3', '_',
                    lhs=B(all(bool(((n+1)*pi/3)^2 < ((n+2)*pi/3)^2) for n in range(10)) and limit(((n_+1)*pi/3)^2, n_=oo) == oo),
                    rhs=1, relation='==', edge=False),
]
