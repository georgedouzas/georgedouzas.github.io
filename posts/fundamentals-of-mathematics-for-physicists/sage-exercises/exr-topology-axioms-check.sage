# Exercise exr-topology-axioms-check: test the topology axioms on three families of subsets of {1, 2, 3}.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _topologies import is_topology
X = [1, 2, 3]; F = frozenset
fams = {"{0,{1},{2,3},X}": [F(), F([1]), F([2, 3]), F(X)], "{0,{1},{1,2},{1,3},X}": [F(), F([1]), F([1, 2]), F([1, 3]), F(X)], "{0,{1,2},{2,3},X}": [F(), F([1, 2]), F([2, 3]), F(X)]}
res = {k: is_topology(v, X) for k, v in fams.items()}
print(res)
print("[PASS] the first two are topologies, the third is not ({2} is missing):", list(res.values()) == [True, True, False])
