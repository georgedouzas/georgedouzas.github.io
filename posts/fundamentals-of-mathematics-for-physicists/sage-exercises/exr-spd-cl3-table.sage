# Exercise exr-spd-cl3-table: the multiplication table of Cl(3,0) realised by the Pauli matrices, and its centre.
import itertools
s = [matrix(QQbar, [[0, 1], [1, 0]]), matrix(QQbar, [[0, -I], [I, 0]]), matrix(QQbar, [[1, 0], [0, -1]])]
subsets = [A for k in range(4) for A in itertools.combinations(range(3), k)]
def sig(A): return prod([s[a] for a in A], identity_matrix(QQbar, 2))
basis = {A: sig(A) for A in subsets}
name = lambda A: "1" if not A else "s" + "".join(str(a + 1) for a in A)
def decompose(M):
    # over the reals the eight sigma_A are independent: look for a real sign first
    for cs in ((1, -1), (I, -I)):
        for A, B in basis.items():
            for c in cs:
                if M == c * B: return c, A
print("each product sigma_A sigma_B is +sigma_C or -sigma_C:", all(decompose(basis[A] * basis[B])[0] in (1, -1) for A in subsets for B in subsets))
def show(c, A):
    sgn = {1: "+", -1: "-", I: "+i ", -I: "-i "}[c]
    return (sgn + name(A)) if A or c in (1, -1) else sgn.strip()
print("      " + " ".join(name(B).rjust(7) for B in subsets))
for A in subsets:
    print(name(A).ljust(5), " ".join(show(*decompose(basis[A] * basis[B])).rjust(7) for B in subsets))
central = [name(A) for A in subsets if all(basis[A] * basis[B] == basis[B] * basis[A] for B in subsets)]
print("central basis elements:", central, "; s1 s2 s3 =", basis[(0, 1, 2)].list())
