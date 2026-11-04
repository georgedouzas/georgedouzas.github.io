# Exercise exr-equivalence-check: test the three axioms for three relations on {1, 2, 3, 4}.
X = [1, 2, 3, 4]
rels = {"same parity": lambda a, b: (a - b) % 2 == 0, "a <= b": lambda a, b: a <= b, "|a - b| <= 1": lambda a, b: abs(a - b) <= 1}
for name, r in rels.items():
    refl = all(r(a, a) for a in X); symm = all(r(a, b) == r(b, a) for a in X for b in X)
    trans = all((not (r(a, b) and r(b, c))) or r(a, c) for a in X for b in X for c in X)
    print("%-14s reflexive %s symmetric %s transitive %s" % (name, refl, symm, trans))
ok = lambda r: all(r(a, a) for a in X) and all(r(a, b) == r(b, a) for a in X for b in X) and all((not (r(a, b) and r(b, c))) or r(a, c) for a in X for b in X for c in X)
print("[PASS] only 'same parity' is an equivalence relation:", [ok(r) for r in rels.values()] == [True, False, False])
