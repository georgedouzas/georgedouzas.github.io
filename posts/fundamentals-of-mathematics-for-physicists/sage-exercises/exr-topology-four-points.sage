# Exercise exr-topology-four-points: the topologies on a four-element set (2^16 families tested).
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _topologies import topologies
T = topologies([1, 2, 3, 4])
print("topologies on four points:", len(T))
print("[PASS] 355 topologies on four points:", len(T) == 355)
