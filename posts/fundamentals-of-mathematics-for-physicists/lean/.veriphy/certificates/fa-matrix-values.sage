# name: fa-matrix-values
# technique: matrix identity read from the goal (matrix_from_goal recipe, adapted): the entries are trigonometric
#   values at 0, π and 2π, which Sage evaluates exactly and Lean evaluates with the value lemmas
# goal: !![…] = 1 or = -1 for an explicit matrix whose entries are cos 0, sin 0, cos π, sin π, cos (2π), sin (2π) and √2
# description: Sage checks that every entry of lhs - rhs is exactly 0 (cos 0 = 1, sin π = 0, …); Lean closes each
#   entry with simp (Real.cos_zero, Real.sin_pi, Real.cos_two_pi, …) and norm_num.
from sage.all import SR
e = goal.expr
assert e is not None, 'the goal has no symbolic form'
entries = [SR(x) for x in e.list()] if hasattr(e, 'nrows') else [SR(e)]
bad = [x for x in entries if not bool(x.simplify_full() == 0)]
assert not bad, f'the identity is false: nonzero entries {bad}'
proof = tactic('ext i j\nfin_cases i <;> fin_cases j <;> simp [Matrix.one_apply, Real.cos_zero, Real.sin_zero, Real.cos_pi, '
               'Real.sin_pi, Real.cos_two_pi, Real.sin_two_pi] <;> norm_num')
