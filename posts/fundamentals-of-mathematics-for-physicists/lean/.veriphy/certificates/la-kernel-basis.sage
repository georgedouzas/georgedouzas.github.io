# name: la-kernel-basis
# technique: matrix computation read from the goal (matrix_from_goal recipe), with the exercise's Sage solution as oracle
# goal: A times the kernel vectors (-2, 1, 0), (-3, 0, 1), (1, 1, -1), (3, 0, -1) is zero; coordinates in the kernel bases
# description: the oracle reruns the chapter's exercise script and asserts its answer; Sage then evaluates lhs - rhs
#   from the goal and Lean proves it entrywise with norm_num, or by the closed det/trace formula.
# oracle (sage/exercises/exr-kernel-basis.sage)
A = matrix(QQ, [[1, 2, 3], [2, 4, 6]])
K = A.right_kernel()
for v in [vector(QQ, [-2, 1, 0]), vector(QQ, [-3, 0, 1]), vector(QQ, [1, 1, -1]), vector(QQ, [3, 0, -1])]:
    assert A * v == 0
assert K.coordinates(vector(QQ, [1, 1, -1])) == [1, 1] and K.coordinates(vector(QQ, [3, 0, -1])) == [3, 0]
print('oracle: echelon kernel basis', K.basis())
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
