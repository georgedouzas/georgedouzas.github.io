# name: set-t0-three
# technique: complete enumeration by Sage as a witness (witness recipe), checked by decide
# goal: ∃ L : List (Finset (Finset (Fin 3))), L.Nodup ∧ L.length = 19 ∧ ∀ F, F ∈ L ↔ IsFinTop F ∧ IsT0Fin F
# description: Sage enumerates the 29 topologies on three points and keeps those in which any two distinct points have
#   an open set containing one and not the other (the solution's test); the kernel checks the list.
X = frozenset([0, 1, 2])
subsets = [frozenset(s) for s in Subsets(X)]
def is_top(T):
    T = set(T)
    return frozenset() in T and X in T and all((A | B) in T and (A & B) in T for A in T for B in T)
def is_t0(T):
    return all(any((x in U) != (y in U) for U in T) for x in X for y in X if x != y)
tops = [T for T in Subsets(subsets) if is_top(T)]
t0 = [T for T in tops if is_t0(T)]
fs = lambda S: '∅' if not S else '{' + ', '.join(map(str, sorted(S))) + '}'
fam = lambda T: '{' + ', '.join(fs(S) for S in sorted(T, key=lambda S: (len(S), sorted(S)))) + '}'
print('Sage:', len(tops), 'topologies,', len(t0), 'of them T0')
proof = tactic('set_option maxRecDepth 100000 in exact ⟨[' + ', '.join(fam(T) for T in t0) + f'], by decide, rfl, by decide⟩')
