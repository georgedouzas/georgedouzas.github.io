# name: set-close-not-trans
# technique: finite witness found by enumeration (witness recipe), checked by decide
# goal: ∃ a b c : ℤ in {1, 2, 3, 4}, |a - b| ≤ 1 ∧ |b - c| ≤ 1 ∧ ¬ |a - c| ≤ 1 (|a - b| ≤ 1 is not transitive)
# description: Sage tests transitivity on all triples of {1, 2, 3, 4} and returns the first failing triple.
X = range(1, 5)
R = lambda a, b: abs(a - b) <= 1
fails = [(a, b, c) for a in X for b in X for c in X if R(a, b) and R(b, c) and not R(a, c)]
a, b, c = fails[0]
print('Sage: transitivity fails at', (a, b, c), 'among', len(fails), 'triples')
proof = tactic(f'exact ⟨{a}, {b}, {c}, by decide⟩')
