# name: set-topology-missing-union
# technique: finite witness found by enumeration (witness recipe), checked by decide
# goal: ∃ U ∈ F, ∃ V ∈ F, U ∪ V ∉ F for F = {∅, {1}, {2}, {1, 2, 3}} (as 0-based Finsets of Fin 3)
# description: Sage tests all pairs of members and returns a pair whose union is missing ({1} ∪ {2} = {1, 2}).
F = [frozenset(), frozenset([0]), frozenset([1]), frozenset([0, 1, 2])]
pairs = [(U, V) for U in F for V in F if (U | V) not in F]
U, V = pairs[0]
fs = lambda S: '∅' if not S else '{' + ', '.join(map(str, sorted(S))) + '}'
print('Sage: missing unions', [(sorted(a), sorted(b)) for a, b in pairs])
proof = tactic(f'exact ⟨{fs(U)}, by decide, {fs(V)}, by decide, by decide⟩')
