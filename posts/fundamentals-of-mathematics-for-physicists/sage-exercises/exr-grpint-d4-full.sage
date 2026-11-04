# Exercise exr-grpint-d4-full: the dihedral group D_4 of the square through the whole chapter: orders, subgroups and
# normal subgroups, the quotient by the centre, the orbit-stabiliser count on the vertices, conjugacy classes, the
# sign of the permutation of the vertices as a homomorphism, and the irreducible representations.
from collections import Counter
D4 = DihedralGroup(4)
print("|D_4| =", D4.order(), "; abelian:", D4.is_abelian(), "; element orders:", dict(sorted(Counter(g.order() for g in D4).items())))
subs = D4.subgroups(); normal = D4.normal_subgroups()
print("subgroups:", len(subs), "of orders", sorted(H.order() for H in subs))
print("normal subgroups:", len(normal), "of orders", sorted(H.order() for H in normal))
print("every subgroup order divides 8 (Lagrange):", all(8 % H.order() == 0 for H in subs))
Z = D4.center(); Q = D4.quotient(Z)
print("centre", [str(z) for z in Z], "; D_4/Z has order", Q.order(), "and is cyclic:", Q.is_cyclic())
orb, stab = D4.orbit(1), D4.stabilizer(1)
print("orbit of vertex 1:", orb, "; stabiliser:", [str(s) for s in stab], "; |orbit| |stab| =", len(orb) * stab.order())
reps = D4.conjugacy_classes_representatives()
print("conjugacy classes:", [[str(g) for g in D4.conjugacy_class(r)] for r in reps])
K = D4.subgroup([g for g in D4 if g.sign() == 1])
print("sign of the vertex permutation: kernel", [str(k) for k in K], "of order", K.order(), "; normal:", K.is_normal(D4))
dims = [chi.degree() for chi in D4.irreducible_characters()]
print("irreducible dimensions", dims, "; sum of squares", sum(d**2 for d in dims), "; number of irreducibles = number of classes:", len(dims) == len(reps))
