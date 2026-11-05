# name: set-union-not-topology
# technique: finite witness found by enumeration of the topologies on three points (witness recipe), checked by decide
# goal: ∃ F G : Finset (Finset (Fin 3)), IsFinTop F ∧ IsFinTop G ∧ ¬ IsFinTop (F ∪ G)
# description: Sage enumerates the 29 topologies on three points and returns the first pair (in the order of the
#   solution: {∅, {1}, X} and {∅, {2}, X} first) whose union fails the axioms.
import itertools
X = frozenset([0, 1, 2])
subsets = [frozenset(s) for s in Subsets(X)]
def is_top(T):
    T = set(T)
    return frozenset() in T and X in T and all((A | B) in T and (A & B) in T for A in T for B in T)
tops = [frozenset(T) for T in Subsets(subsets) if is_top(T)]
pref = [frozenset([frozenset(), frozenset([0]), X]), frozenset([frozenset(), frozenset([1]), X])]
pairs = [(F, G) for F in tops for G in tops if not is_top(F | G)]
F, G = tuple(pref) if tuple(pref) in pairs else pairs[0]
fs = lambda S: '∅' if not S else '{' + ', '.join(map(str, sorted(S))) + '}'
fam = lambda T: '{' + ', '.join(fs(S) for S in sorted(T, key=lambda S: (len(S), sorted(S)))) + '}'
print('Sage:', len(tops), 'topologies;', len(pairs), 'pairs with a union that is not a topology; witness', fam(F), fam(G))
proof = tactic(f'exact ⟨{fam(F)}, {fam(G)}, by decide, by decide, by decide⟩')
