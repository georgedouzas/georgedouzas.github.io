# name: ra-rearranged-blocks
# technique: closed-form sum by induction, with denominators cleared in the step
# goal: rearranged block sums = ½ ∑_{k<2b} (-1)^k/(k+1)
# description: Induction on n (sum_induction); the step clears the denominators (field_simp) before ring.
# the summand is a named definition (blockTerm, altTerm): a sum whose summand divides by (or takes a root of) the bound
# variable is not translated by Veriphy 0.3.0 ("unknown free variable"); Lean unfolds the name in the induction step
import re
found = re.search(r'(?:Finset\.)?range \(?(\w+)', goal.text if goal is not None else '')
assert found, 'the goal must be a sum over range n'
n = found.group(1)
proof = tactic(f'induction {n} with\n| zero => simp [blockTerm, altTerm]\n| succ {n} ih =>\n  rw [Finset.sum_range_succ, ih, show 2 * ({n} + 1) = 2 * {n} + 1 + 1 by ring, Finset.sum_range_succ, Finset.sum_range_succ]\n  simp only [blockTerm, altTerm]\n  have e : (-1 : ℝ) ^ (2 * {n}) = 1 := by rw [pow_mul]; norm_num\n  rw [pow_succ, e]\n  push_cast\n  field_simp\n  ring')
