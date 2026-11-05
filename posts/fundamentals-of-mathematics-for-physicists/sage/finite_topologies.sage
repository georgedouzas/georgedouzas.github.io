# Oracle: all topologies on a three-point set by exhaustive check of the axioms on every family of subsets (Sage's
# Subsets), and which of them are Hausdorff (distinct points have disjoint open neighbourhoods). Evidence for
# prp-finite-topologies and prp-hausdorff-topologies.
X = frozenset([1, 2, 3])
subsets = [frozenset(s) for s in Subsets(X)]
def is_topology(T):
    T = set(T)
    if frozenset() not in T or X not in T: return False
    return all((A | B) in T and (A & B) in T for A in T for B in T)
def is_hausdorff(T):
    return all(any(x in U and y in V and not (U & V) for U in T for V in T) for x in X for y in X if x != y)
tops = [T for T in Subsets(subsets) if is_topology(T)]
haus = [T for T in tops if is_hausdorff(T)]
print("[PASS] number of topologies on a 3-point set: 29:", len(tops) == 29)
print("[PASS] exactly one topology on a 3-point set is Hausdorff, the discrete one:", len(haus) == 1 and len(haus[0]) == 8)
