# name: la-tensor-components
# technique: matrix computation read from the goal (matrix_from_goal recipe), with the exercise's Sage solution as oracle
# goal: contraction a(u) v = (-2, 0, 2), components of u and u ⊗ v in the basis f = (e₁ + e₂, e₂, e₃), and the trace of T in both bases
# description: the oracle reruns the chapter's exercise script and asserts its answer; Sage then evaluates lhs - rhs
#   from the goal and Lean proves it entrywise with norm_num, or by the closed det/trace formula.
# oracle (sage/exercises/exr-tensor-components.sage)
V = FiniteRankFreeModule(QQ, 3, name='V'); e = V.basis('e')
u = V([1, 2, 3], basis=e); v = V([1, 0, -1], basis=e); t = u * v
a = V.linear_form(); a[:] = [1, 0, -1]
assert a(u) == -2 and list(a.contract(t).comp(e)[:]) == [-2, 0, 2]
f = V.basis('f', from_family=(e[0] + e[1], e[1], e[2]))
assert list(u.comp(f)[:]) == [1, 1, 3]
T = V.tensor((1, 1)); T[:] = [[1, 2, 0], [0, 3, 1], [0, 0, 5]]
assert T.trace() == 9 == sum(T.comp(f)[i, i] for i in range(3))
print('oracle: T in f =', [[T.comp(f)[i, j] for j in range(3)] for i in range(3)], '; u (x) v in f =', [[t.comp(f)[i, j] for j in range(3)] for i in range(3)])
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
