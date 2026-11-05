# name: set-topologies-two
# technique: complete enumeration by Sage as a witness (witness recipe), checked by decide
# goal: ∃ L : List (Finset (Finset (Fin 2))), L.Nodup ∧ L.length = 4 ∧ ∀ F, F ∈ L ↔ IsFinTop F
# description: Sage enumerates the 2^4 = 16 families of subsets of {1, 2} and keeps the topologies (the solution's
#   enumeration); the kernel checks that the list has no repetition, its length, and that it is exactly the set of
#   families satisfying the axioms.
X = frozenset([0, 1])
subsets = [frozenset(s) for s in Subsets(X)]
def is_top(T):
    T = set(T)
    return frozenset() in T and X in T and all((A | B) in T and (A & B) in T for A in T for B in T)
tops = [T for T in Subsets(subsets) if is_top(T)]
fs = lambda S: '∅' if not S else '{' + ', '.join(map(str, sorted(S))) + '}'
fam = lambda T: '{' + ', '.join(fs(S) for S in sorted(T, key=lambda S: (len(S), sorted(S)))) + '}'
print('Sage:', len(subsets), 'subsets,', 2 ** len(subsets), 'families,', len(tops), 'topologies:', [fam(T) for T in tops])
proof = tactic('set_option maxRecDepth 100000 in exact ⟨[' + ', '.join(fam(T) for T in tops) + f'], by decide, rfl, by decide⟩')
