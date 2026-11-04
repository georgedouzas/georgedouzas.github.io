# Exercise exr-topology-t0-count: the T0 topologies among the 29 on three points.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _topologies import topologies
X = [1, 2, 3]; T = topologies(X)
def t0(t):
    return all(any((a in U) != (b in U) for U in t) for a in X for b in X if a < b)
n0 = sum(1 for t in T if t0(t))
print("topologies:", len(T), "; T0 among them:", n0)
print("[PASS] 19 of the 29 topologies are T0:", (len(T), n0) == (29, 19))
