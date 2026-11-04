# Exercise exr-finite-topologies: all topologies on a three-point set, found by checking the axioms on every family
# of subsets (Sage's Subsets).
X = frozenset([1, 2, 3])
subsets = [frozenset(s) for s in Subsets(X)]
def is_topology(T):
    T = set(T)
    if frozenset() not in T or X not in T: return False
    for A in T:
        for B in T:
            if (A | B) not in T or (A & B) not in T: return False
    return True
tops = [T for T in Subsets(subsets) if is_topology(T)]
print("number of topologies on a 3-point set:", len(tops))
print("the coarsest (indiscrete) topology:", sorted([sorted(s) for s in min(tops, key=len)]))
print("the finest (discrete) topology has", len(max(tops, key=len)), "open sets = 2^3")
def is_hausdorff(T):
    return all(any(x in U and y in V and not (U & V) for U in T for V in T) for x in X for y in X if x != y)
hausdorff = [T for T in tops if is_hausdorff(T)]
print("Hausdorff topologies among them (distinct points have disjoint open neighbourhoods):", len(hausdorff), " -> only the discrete one")
