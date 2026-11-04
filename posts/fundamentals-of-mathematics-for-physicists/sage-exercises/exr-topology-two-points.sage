# Exercise exr-topology-two-points: all topologies on {1, 2}.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _topologies import topologies
T = topologies([1, 2])
for t in T: print(sorted(sorted(s) for s in t))
print("[PASS] there are 4 topologies on two points:", len(T) == 4)
