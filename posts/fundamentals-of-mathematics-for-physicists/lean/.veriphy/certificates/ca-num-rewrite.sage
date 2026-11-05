# name: ca-num-rewrite
# technique: rewrite with named mathlib lemmas, checked in Sage first (transcendental_rewrite recipe)
# goal: an identity between values of Complex.arg, Complex.log, Complex.exp, cos and sin at 1, -1, ±i, π, iπ, 2πi, iπ/2
#   (with the project's translation of Complex.arg as Sage's arg)
# description: Sage evaluates both sides exactly (principal branches) and confirms the identity; Lean rewrites with
#   the special-value lemmas and closes the arithmetic with ring or norm_num.
LEMMAS = ['Complex.arg_one', 'Complex.arg_I', 'Complex.arg_neg_one', 'Complex.arg_neg_I', 'Complex.log_neg_one',
          'Complex.log_I', 'Complex.log_neg_I', 'Complex.log_one', 'Complex.exp_pi_mul_I',
          'Complex.exp_two_pi_mul_I', 'Complex.exp_pi_div_two_mul_I', 'Complex.exp_zero', 'Complex.exp_add',
          'Real.cos_pi', 'Real.sin_pi', 'neg_neg']
e = goal.expr
assert e is not None, 'the goal has no symbolic form'
assert bool(e.simplify_full() == 0) or bool(e.simplify_exp().simplify_full() == 0), \
    f'Sage cannot confirm the identity: lhs - rhs = {e.simplify_full()}'
proof = tactic('simp only [' + ', '.join(LEMMAS) + ']\nall_goals first | ring | norm_num')
