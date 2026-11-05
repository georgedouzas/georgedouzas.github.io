# name: matrix_from_goal
# technique: matrix computation read from the goal, checked entrywise
# goal: an identity, determinant or trace of explicit matrices, written in the goal (!![…], *, ⁻¹, det, trace)
# description: Sage evaluates lhs - rhs from the goal itself; Lean proves it entry by entry or by the det/trace formula.
import re
SIMP = []  # EDIT: extra simp lemmas for norm_num, for example definitions of named matrices
e = goal.expr
assert e is not None, 'the goal has no symbolic form; transcribe the matrices with the matrix_identity recipe'
simp = f'norm_num [{", ".join(SIMP)}]' if SIMP else 'norm_num'
if hasattr(e, 'nrows'):
    # a matrix identity: lhs - rhs must be the zero matrix
    assert e.simplify_full() == 0, f'the identity is false: lhs - rhs = {matrix_term(e.simplify_full(), problem)}'
    proof = tactic(f'ext i j\nfin_cases i <;> fin_cases j <;> {simp}')
else:
    # a determinant or trace: lhs - rhs must simplify to 0; rewrite by the formula for the literal's size
    assert bool(e.simplify_full() == 0), f'the identity is false: lhs - rhs = {symbolic(e.simplify_full(), problem)}'
    literal = re.search(r'!!\[(.*?)\]', goal.text)
    assert literal, 'no !![…] literal in the goal; use the matrix_identity recipe'
    n = literal.group(1).count(';') + 1
    assert n in (2, 3), f'no closed formula for {n}×{n}; expand with Matrix.det_succ_row_zero'
    kind = 'det' if 'det' in goal.text else 'trace'
    formula = {('det', 2): 'Matrix.det_fin_two_of', ('det', 3): 'Matrix.det_fin_three',
               ('trace', 2): 'Matrix.trace_fin_two_of', ('trace', 3): 'Matrix.trace_fin_three_of'}[kind, n]
    proof = tactic(f'rw [{formula}]\nring')
