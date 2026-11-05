# name: ra-multivariategaussian-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.MultipleIntegrals.MultivariateGaussian file "ra-multivariategaussian-test.sage"
# description: Sage integrates e^(-xᵀAx/2) over ℝ² exactly for positive definite 2 × 2 matrices and compares with 2π/√det A
SEED = 0  # the instances are listed, no sampling
R = RealIntervalField(128)
def B(b):
    return QQ(1 if bool(b) else 0)
RI = RealIntervalField(256)
def O(values, lhs, rhs, rel='==', edge=False):
    # symbolic values (π, e, logs, limits Sage computed) are compared by 256-bit intervals; rationals exactly
    if any(hasattr(v, 'parent') and v.parent() is SR for v in (lhs, rhs)):
        lhs, rhs = RI(lhs), RI(rhs)
    return oracle_instance(*values, lhs=lhs, rhs=rhs, relation=rel, edge=edge)
x, y = var('x y')
instances = []
for A, edge in [([[1, 0], [0, 1]], True), ([[2, 1], [1, 2]], False), ([[3, -1], [-1, 1]], False)]:
    M = matrix(QQ, A)
    q = (M[0, 0] * x ** 2 + 2 * M[0, 1] * x * y + M[1, 1] * y ** 2) / 2
    val = integrate(integrate(exp(-q), x, -oo, oo), y, -oo, oo)
    instances.append(O(['2', f'!![{A[0][0]}, {A[0][1]}; {A[1][0]}, {A[1][1]}]', '‹_›'], val, 2 * pi / sqrt(M.det()), edge=edge))
