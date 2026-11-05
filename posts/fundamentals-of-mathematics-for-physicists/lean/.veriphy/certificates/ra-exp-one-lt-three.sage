# name: ra-exp-one-lt-three
# technique: rational bounds on transcendental values from mathlib's decimal lemmas or truncated series
# goal: Real.exp 1 < 3
# description: Sage's interval arithmetic picks mathlib's decimal lemma or the truncation order; Lean checks the rational remainder.
import re
text = (goal.text if goal is not None else '').replace('rexp', 'Real.exp')
found = re.search(r'Real\.(\w+)\s*\(?\s*([-0-9./ ]*?)\s*\)?\s*(<|>)\s*([0-9.]+)$', text) or \
    re.search(r'^([0-9.]+)\s*(<|>)\s*Real\.(\w+)\s*\(?\s*([-0-9./ ]*?)\s*\)?$', text)
assert found, f'the goal must compare Real.f c with a decimal: {text}'
if text.startswith('Real.'):
    function, point, relation, bound = found.groups()
    upper = relation == '<'
else:
    bound, relation, function, point = found.groups()
    upper = relation == '>'
point = QQ(sage_eval(point)) if point.strip() else QQ(0)
proof = bound_proof(function.lower(), point, QQ(Integer(bound.replace('.', ''))) / 10 ** len(bound.partition('.')[2]), upper=upper)
