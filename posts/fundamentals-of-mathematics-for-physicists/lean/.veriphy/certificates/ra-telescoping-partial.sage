# name: ra-telescoping-partial
# technique: closed-form sum by induction, with denominators cleared in the step
# goal: ∑_{k<n} 1/((k+1)(k+2)) = 1 - 1/(n+1)
# description: Induction on n (sum_induction); the step clears the denominators (field_simp) before ring.
# the summand is a named definition (telescopingTerm): a sum whose summand divides by (or takes a root of) the bound
# variable is not translated by Veriphy 0.3.0 ("unknown free variable"); Lean unfolds the name in the induction step
import re
found = re.search(r'(?:Finset\.)?range \(?(\w+)', goal.text if goal is not None else '')
assert found, 'the goal must be a sum over range n'
n = found.group(1)
proof = tactic(f'induction {n} with\n| zero => simp [telescopingTerm]\n| succ {n} ih =>\n  rw [Finset.sum_range_succ, ih]\n  simp only [telescopingTerm]\n  push_cast\n  field_simp\n  ring')
