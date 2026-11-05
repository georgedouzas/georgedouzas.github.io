# name: ca-res-sum
# technique: closed-form sum by induction (sum_induction recipe)
# goal: ∑ k ∈ Finset.range n, f k = F n (or over Finset.Icc 1 n, Finset.Ico 1 n), with the closed form F stated
# description: Induction on n; both cases close by push_cast and ring1. Sage only finds F (see the recipe).
import re
found = re.search(r'Finset\.(range|Icc|Ico) (?:(\d+) )?(\w+)', goal.text if goal is not None else '')
assert found, 'the goal must be a sum over Finset.range n, Finset.Icc a n or Finset.Ico a n'
index_set, lower, n = found.group(1), int(found.group(2) or 0), found.group(3)
proof = sum_by_induction(n, index_set, lower)
