# name: ra-sqrt-telescoping
# technique: closed-form sum by induction
# goal: ∑_{k<n} (√(k+2) - √(k+1)) = √(n+1) - 1
# description: Induction on n; both cases close by push_cast and ring1. Sage only finds the closed form.
# the summand is a named definition (sqrtStep): a sum whose summand divides by (or takes a root of) the bound
# variable is not translated by Veriphy 0.3.0 ("unknown free variable"); Lean unfolds the name in the induction step
import re
found = re.search(r'(?:Finset\.)?range \(?(\w+)', goal.text if goal is not None else '')
assert found, 'the goal must be a sum over range n'
n = found.group(1)
proof = tactic(f'induction {n} with\n| zero => simp [sqrtStep]\n| succ {n} ih =>\n  rw [Finset.sum_range_succ, ih]\n  simp only [sqrtStep]\n  push_cast\n  ring_nf')
