# Exercise exr-bijection-count-4: the 24 permutations of {1,2,3,4} and the 6 that fix 4.
G = SymmetricGroup(4)
fix4 = [g for g in G if g(4) == 4]
print("order of S_4:", G.order()); print("permutations fixing 4:", [str(g) for g in fix4])
print("[PASS] 24 bijections, 6 fixing 4:", G.order() == 24 and len(fix4) == 6)
