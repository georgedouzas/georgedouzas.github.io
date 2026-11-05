# name: set-perm-inverse
# technique: finite witness found with Sage's SymmetricGroup (witness recipe), checked by decide
# goal: ∃ u : Equiv.Perm (Fin 3), cycleT * u = 1 ∧ u * cycleT = 1 ∧ u 0 = 2 ∧ u 2 = 1 ∧ u 1 = 0
# description: Sage inverts t = (1 2 3), giving (1 3 2); the witness is that permutation with 0-based labels, and the
#   kernel checks both products and its values.
G = SymmetricGroup(3)
t = G('(1,2,3)')
u = t.inverse()
print('Sage: t^-1 =', u, '; t * t^-1 =', t * u)
fwd = [u(i) - 1 for i in (1, 2, 3)]
bwd = [t(i) - 1 for i in (1, 2, 3)]
proof = tactic(f'exact ⟨⟨![{", ".join(map(str, fwd))}], ![{", ".join(map(str, bwd))}], by decide, by decide⟩, by decide⟩')
