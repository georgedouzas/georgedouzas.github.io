# Exercise exr-hausdorff-count-t1: topologies on three points in which every point is closed.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _topologies import topologies
X = [1, 2, 3]; T = topologies(X)
t1 = [t for t in T if all(frozenset(set(X) - {p}) in t for p in X)]
print("topologies with all points closed:", len(t1))
print("[PASS] exactly one (the discrete topology):", len(t1) == 1 and len(t1[0]) == 8)
