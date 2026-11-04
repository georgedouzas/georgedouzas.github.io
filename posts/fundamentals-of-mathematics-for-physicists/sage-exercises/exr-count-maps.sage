# Exercise exr-count-maps: counting maps, injections and bijections between finite sets (Sage's combinatorics).
A = Set([1, 2, 3]); B = Set(["x", "y"])
maps = [dict(zip(A, vals)) for vals in cartesian_product([list(B)] * len(A))]
print("maps A -> B:", len(maps), "= 2^3")
injective = [m for m in maps if len(set(m.values())) == len(A)]
surjective = [m for m in maps if set(m.values()) == set(B)]
print("injective maps A -> B:", len(injective), " (none: |A| > |B|);  surjective maps:", len(surjective), "= 2^3 - 2")
C = Set(["x", "y", "z"])
bij = [p for p in Permutations(list(C))]
print("bijections C -> C:", len(bij), "= 3!")
