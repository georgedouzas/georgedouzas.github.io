# Exercise exr-product-count: Cartesian products of finite sets.
P = cartesian_product([[1, 2], ["a", "b", "c"]]); Q = cartesian_product([[1, 2], ["a", "b", "c"], ["x", "y"]])
print(list(P)); print("|{1,2} x {a,b,c} x {x,y}| =", Q.cardinality())
print("[PASS] 6 pairs and 12 triples:", P.cardinality() == 6 and Q.cardinality() == 12)
