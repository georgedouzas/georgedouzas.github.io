# name: pr-power-sum
# technique: closed-form sum by induction
# goal: ∑ k ∈ Finset.Icc 1 n, k^m = F n over ℚ (the power sums behind the moments of a fair die), or a sum over
#   Finset.range n
# description: Sage checks the closed form against the summed polynomial; induction on n, both cases close by
#   push_cast and ring1 (sum_induction recipe).
import re
found = re.search(r'(?:Finset\.)?(range|Icc|Ico) (?:(\d+) )?(\w+)', goal.text if goal is not None else '')
assert found, 'the goal must be a sum over Finset.range n, Finset.Icc a n or Finset.Ico a n'
index_set, lower, n = found.group(1), int(found.group(2) or 0), found.group(3)
proof = sum_by_induction(n, index_set, lower)
# oracle: Sage evaluates the sum read off the goal symbolically and checks the stated closed form
if goal.expr is not None:
    assert bool(goal.expr.simplify_full() == 0), f'the closed form is wrong: lhs - rhs = {goal.expr.simplify_full()}'
