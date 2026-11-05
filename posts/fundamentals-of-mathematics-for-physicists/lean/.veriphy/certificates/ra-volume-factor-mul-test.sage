# name: ra-volume-factor-mul-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Calculus.volume_factor_mul file "ra-volume-factor-mul-test.sage"
# description: det(AB) = det A det B, exactly at sample rational matrices
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
instances = []
import itertools
for A, B_, edge in [([[1, 2], [3, 4]], [[0, 1], [1, 0]], False), ([[2, 0], [0, 3]], [[QQ(1) / 2, 1], [-1, 5]], False), ([[0, 0], [0, 0]], [[1, 2], [3, 4]], True)]:
    MA, MB = matrix(QQ, A), matrix(QQ, B_)
    lean = lambda M: '!![' + '; '.join(', '.join(str(c) for c in row) for row in M.rows()) + ']'
    instances.append(O([f'({lean(MA)} : Matrix (Fin 2) (Fin 2) ℚ)', f'({lean(MB)} : Matrix (Fin 2) (Fin 2) ℚ)'], (MA * MB).det(), MA.det() * MB.det(), edge=edge))
