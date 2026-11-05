# name: set-zmod6-double-nonsurjective
# technique: finite witness found by enumeration in Zmod(6) (witness recipe), checked by decide
# goal: ∃ y : ZMod 6, ∀ k : ZMod 6, 2 * k ≠ y (doubling on ℤ₆ is not surjective)
# description: Sage computes the image {0, 2, 4} of doubling and returns the least element outside it.
R = Zmod(6)
image = set(2 * k for k in R)
y = [v for v in R if v not in image][0]
print('Sage: image of doubling', sorted(image), '; missed', y)
proof = tactic(f'exact ⟨{y}, by decide⟩')
