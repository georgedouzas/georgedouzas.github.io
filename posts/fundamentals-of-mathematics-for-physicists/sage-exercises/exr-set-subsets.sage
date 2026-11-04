# Exercise exr-set-subsets: all subsets of {a, b, c}.
X = Set(["a", "b", "c"])
subs = list(X.subsets())
for s in subs: print(sorted(s))
print("[PASS] the number of subsets is 2^3 = 8:", len(subs) == 8)
