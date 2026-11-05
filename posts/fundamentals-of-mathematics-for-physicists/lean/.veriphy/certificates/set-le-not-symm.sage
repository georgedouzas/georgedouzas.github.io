# name: set-le-not-symm
# technique: finite witness found by enumeration (witness recipe), checked by decide
# goal: ∃ a b : ℤ, 1 ≤ a ∧ a ≤ 4 ∧ 1 ≤ b ∧ b ≤ 4 ∧ a ≤ b ∧ ¬ b ≤ a (≤ is not symmetric on {1, 2, 3, 4})
# description: Sage tests symmetry on all pairs of {1, 2, 3, 4} and returns the first failing pair.
X = range(1, 5)
fails = [(a, b) for a in X for b in X if a <= b and not b <= a]
a, b = fails[0]
print('Sage: symmetry fails at', (a, b), 'among', len(fails), 'pairs')
proof = tactic(f'exact ⟨{a}, {b}, by decide⟩')
