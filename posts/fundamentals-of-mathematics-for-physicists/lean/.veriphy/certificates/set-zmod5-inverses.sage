# name: set-zmod5-inverses
# technique: finite witness found by Sage's inverses in Zmod(5) (witness recipe), checked by decide
# goal: ∃ b₁ b₂ b₃ b₄ : ZMod 5, 1 * b₁ = 1 ∧ 2 * b₂ = 1 ∧ 3 * b₃ = 1 ∧ 4 * b₄ = 1
# description: Sage inverts the non-zero classes of Zmod(5); the kernel checks the four products.
R = Zmod(5)
inv = [R(a) ** -1 for a in (1, 2, 3, 4)]
print('Sage: inverses of 1, 2, 3, 4 modulo 5:', inv)
proof = tactic('exact ⟨' + ', '.join(str(b) for b in inv) + ', by decide⟩')
