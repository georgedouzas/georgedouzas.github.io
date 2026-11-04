# Exercise exr-hausdorff-check-finite: the Hausdorff topologies among those on two and three points.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _topologies import topologies
def hausdorff(t, X):
    return all(any(a in U and b in V and not (U & V) for U in t for V in t) for a in X for b in X if a < b)
res = {}
for X in ([1, 2], [1, 2, 3]):
    T = topologies(X); h = [t for t in T if hausdorff(t, X)]
    res[len(X)] = (len(T), len(h), all(len(t) == 2**len(X) for t in h))
    print("points:", len(X), "topologies:", len(T), "Hausdorff:", len(h), "(discrete)" if res[len(X)][2] else "")
print("[PASS] exactly one Hausdorff topology, the discrete one, on two and on three points:", res == {2: (4, 1, True), 3: (29, 1, True)})
