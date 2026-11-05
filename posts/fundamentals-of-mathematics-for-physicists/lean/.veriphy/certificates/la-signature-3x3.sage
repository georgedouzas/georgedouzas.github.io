# name: la-signature-3x3
# technique: matrix computation read from the goal (matrix_from_goal recipe), with the exercise's Sage solution as oracle
# goal: Pᵀ B P = diag(2, 3/2, 4/3) and Qᵀ C Q = diag(2, -2, -1): rational congruences giving the signatures (3, 0, 0) and (1, 2, 0)
# description: the oracle reruns the chapter's exercise script and asserts its answer; Sage then evaluates lhs - rhs
#   from the goal and Lean proves it entrywise with norm_num, or by the closed det/trace formula.
# oracle (sage/exercises/exr-signature-3x3.sage)
B = matrix(QQ, [[2, 1, 0], [1, 2, 1], [0, 1, 2]]); C = matrix(QQ, [[0, 1, 0], [1, 0, 0], [0, 0, -1]])
assert QuadraticForm(QQ, 2 * B).signature_vector() == (3, 0, 0) and QuadraticForm(QQ, 2 * C).signature_vector() == (1, 2, 0)
P = matrix(QQ, [[1, -1/2, 1/3], [0, 1, -2/3], [0, 0, 1]]); assert P.transpose() * B * P == diagonal_matrix(QQ, [2, 3/2, 4/3])
Q = matrix(QQ, [[1, 1, 0], [1, -1, 0], [0, 0, 1]]); assert Q.transpose() * C * Q == diagonal_matrix(QQ, [2, -2, -1])
print('oracle: eigenvalues of B', B.charpoly().roots(QQbar), '; of C', C.charpoly().roots(QQ))
import re
SIMP = ['Matrix.mul_apply', 'Fin.sum_univ_succ', 'Matrix.transpose_apply']
e = goal.expr
assert e is not None, 'the goal has no symbolic form; transcribe the matrices with the matrix_identity recipe'
simp = f'norm_num [{", ".join(SIMP)}]'
if hasattr(e, 'nrows'):
    assert e.simplify_full() == 0, f'the identity is false: lhs - rhs = {matrix_term(e.simplify_full(), problem)}'
    symbolic_entries = set(re.findall(r"(?<![\w.])[^\W\d][\w']*", goal.text)) - {'Matrix', 'det', 'trace', 'ᵀ'}
    close = ' <;> ring1' if symbolic_entries else ''  # symbolic entries: norm_num leaves linear identities to ring1
    proof = tactic(f'ext i j\nfin_cases i <;> fin_cases j <;> {simp}{close}')
else:
    assert bool(e.simplify_full() == 0), f'the identity is false: lhs - rhs = {symbolic(e.simplify_full(), problem)}'
    literal = re.search(r'!!\[(.*?)\]', goal.text)
    assert literal, 'no !![…] literal in the goal; use the matrix_identity recipe'
    n = literal.group(1).count(';') + 1
    kind = 'det' if 'det' in goal.text else 'trace'
    formula = {('det', 2): 'Matrix.det_fin_two_of', ('det', 3): 'Matrix.det_fin_three',
               ('det', 4): 'Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove',
               ('trace', 2): 'Matrix.trace_fin_two_of', ('trace', 3): 'Matrix.trace_fin_three_of'}[kind, n]
    proof = tactic(f'norm_num [{formula}]')  # 4×4: Laplace expansion along the first row (no closed formula)
