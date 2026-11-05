"""Shared enumeration of the topologies on a small finite set (used by several exercises)."""
from itertools import combinations
def subsets(X):
    out = []
    for r in range(len(X) + 1):
        out += [frozenset(c) for c in combinations(X, r)]
    return out
def is_topology(fam, X):
    fam = set(fam); X = frozenset(X)
    if frozenset() not in fam or X not in fam: return False
    for A in fam:
        for B in fam:
            if (A & B) not in fam or (A | B) not in fam: return False
    return True
def topologies(X):
    subs = subsets(X); n = len(subs)
    out = []
    for mask in range(1 << n):
        fam = [subs[i] for i in range(n) if mask >> i & 1]
        if is_topology(fam, X): out.append(frozenset(fam))
    return out
