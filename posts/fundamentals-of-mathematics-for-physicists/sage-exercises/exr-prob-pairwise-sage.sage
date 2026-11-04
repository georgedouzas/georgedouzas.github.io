# Exercise exr-prob-pairwise-sage: pairwise versus mutual independence. Two coins: A = first heads, B = second heads,
# C = the two agree; three coins: A_i = toss i heads, D = even number of heads.
import itertools
def check(Omega, events, names):
    P = lambda E: QQ(len(E)) / len(Omega)
    for r in range(2, len(events) + 1):
        for idx in itertools.combinations(range(len(events)), r):
            inter = [w for w in Omega if all(w in events[i] for i in idx)]
            ok = P(inter) == prod(P(events[i]) for i in idx)
            print("  ", " & ".join(names[i] for i in idx), ": P(intersection) =", P(inter), ", product =", prod(P(events[i]) for i in idx), ", independent:", ok)
two = list(itertools.product("HT", repeat=2))
A = [w for w in two if w[0] == "H"]; B = [w for w in two if w[1] == "H"]; C = [w for w in two if w[0] == w[1]]
print("two coins:")
check(two, [A, B, C], ["A", "B", "C"])
three = list(itertools.product("HT", repeat=3))
A1 = [w for w in three if w[0] == "H"]; A2 = [w for w in three if w[1] == "H"]; A3 = [w for w in three if w[2] == "H"]
D = [w for w in three if w.count("H") % 2 == 0]
print("three coins:")
check(three, [A1, A2, A3, D], ["A1", "A2", "A3", "D"])
