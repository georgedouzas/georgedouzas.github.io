# name: ca-num-sqrt
# technique: rewrite with a named mathlib lemma, checked in Sage first (transcendental_rewrite recipe)
# goal: √a = b, √a = √b or √a = √b * √c, for explicit rationals a, b, c (b ≥ 0)
# description: Sage confirms the value of the square root exactly; Lean writes b as √(b²) (Real.sqrt_sq), or merges
#   √b √c with Real.sqrt_mul, and norm_num compares the radicands.
e = goal.expr
assert e is not None, 'the goal has no symbolic form'
assert bool(e.simplify_full() == 0) or bool(e.canonicalize_radical() == 0), \
    f'Sage cannot confirm the value: lhs - rhs = {e.simplify_full()}'
rhs_text = goal.text.split(' = ', 1)[1].strip()
if rhs_text.count('√') >= 2:
    proof = tactic('rw [← Real.sqrt_mul (by norm_num)]\nnorm_num')
elif '√' in rhs_text:
    proof = tactic('norm_num')
else:
    proof = tactic(f'conv_rhs => rw [show ({rhs_text} : ℝ) = √(({rhs_text}) ^ 2) from (Real.sqrt_sq (by norm_num)).symm]\n'
                   'norm_num')
