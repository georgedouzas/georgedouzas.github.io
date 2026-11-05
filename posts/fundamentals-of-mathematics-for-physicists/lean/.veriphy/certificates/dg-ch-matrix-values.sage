# name: dg-ch-matrix-values
# technique: matrix identity read from the goal (matrix_from_goal recipe, adapted): the entries are trigonometric
#   values at special angles, which Sage evaluates exactly and Lean evaluates with the value lemmas
# goal: an identity between !![…] matrices whose entries are cos, sin at 0, π/2, π (the Gram matrix of SU(2) at
#   θ = π/2, the Gram matrix of the sphere at special latitudes)
# description: Sage checks that every entry of lhs - rhs is exactly 0 (cos (π/2) = 0, …); Lean closes each entry with
#   simp (Real.cos_pi_div_two, …) and norm_num.
from sage.all import SR
e = goal.expr
assert e is not None, 'the goal has no symbolic form'
entries = [SR(x) for x in e.list()] if hasattr(e, 'nrows') else [SR(e)]
bad = [x for x in entries if not bool(x.simplify_full() == 0)]
assert not bad, f'the identity is false: nonzero entries {bad}'
proof = tactic('ext i j\nfin_cases i <;> fin_cases j <;> simp [Matrix.one_apply, Matrix.smul_apply, Real.cos_zero, Real.sin_zero, '
               'Real.cos_pi, Real.sin_pi, Real.cos_pi_div_two, Real.sin_pi_div_two, Real.cos_pi_div_six, Real.sin_pi_div_six] <;> norm_num')
