# name: ra-geom-two
# technique: closed-form sum by induction
# goal: ∑_{k<n} 2^k = 2^n - 1
# description: Induction on n; both cases close by push_cast and ring1. Sage only finds the closed form.
import re
found = re.search(r'Finset\.(range|Icc|Ico) (?:(\d+) )?(\w+)', goal.text if goal is not None else '')
assert found, 'the goal must be a sum over Finset.range n, Finset.Icc a n or Finset.Ico a n'
index_set, lower, n = found.group(1), int(found.group(2) or 0), found.group(3)
proof = sum_by_induction(n, index_set, lower)
