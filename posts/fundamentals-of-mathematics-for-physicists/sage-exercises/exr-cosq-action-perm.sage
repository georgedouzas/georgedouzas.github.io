# Exercise exr-cosq-action-perm: S_3 acting on the three left cosets of H = <(1 2)> by g.(xH) = (gx)H.
# Sage composes permutations left to right, so we write the text's product g x (first x, then g) as x*g in Sage.
S3 = SymmetricGroup(3); H = [S3.one(), S3("(1,2)")]
def mul(g, x):          # the text's product g x (right-to-left composition)
    return x * g
cosets = []
for x in S3:
    c = frozenset(mul(x, h) for h in H)
    if c not in cosets:
        cosets.append(c)
name = {c: "C%d" % i for i, c in enumerate(cosets)}
print("left cosets xH:", {name[c]: sorted(str(y) for y in c) for c in cosets})
def act(g, c):
    return frozenset(mul(g, y) for y in c)
perm = {g: tuple(cosets.index(act(g, c)) for c in cosets) for g in S3}
for g in S3:
    print("  g = %-8s permutes (C0, C1, C2) to %s" % (g, tuple(name[cosets[i]] for i in perm[g])))
print("identity acts trivially:", perm[S3.one()] == (0, 1, 2))
print("g.(h.c) = (gh).c for all g, h, c:", all(act(g, act(h, c)) == act(mul(g, h), c) for g in S3 for h in S3 for c in cosets))
print("the six permutations are distinct:", len(set(perm.values())) == 6)
