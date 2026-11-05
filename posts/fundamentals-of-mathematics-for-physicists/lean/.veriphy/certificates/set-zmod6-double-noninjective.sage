# name: set-zmod6-double-noninjective
# technique: finite witness found by enumeration in Zmod(6) (witness recipe), checked by decide
# goal: ∃ a b : ZMod 6, a ≠ b ∧ 2 * a = 2 * b (doubling on ℤ₆ is not injective)
# description: Sage enumerates the pairs of Zmod(6) and returns the first distinct pair with equal doubles.
R = Zmod(6)
pairs = [(a, b) for a in R for b in R if a != b and 2 * a == 2 * b]
a, b = pairs[0]
print('Sage: 2 *', a, '= 2 *', b, '=', 2 * a, '; image of doubling', sorted(set(2 * k for k in R)))
proof = tactic(f'exact ⟨{a}, {b}, by decide⟩')
