# Exercise exr-cosq-quotient-d4: the quotient of D_4 by its centre has order 4 and is not cyclic (it is Z_2 x Z_2).
D4 = DihedralGroup(4); Z = D4.center()
Q = D4.quotient(Z)
print("centre of D_4:", [str(z) for z in Z])
print("D_4/Z has order", Q.order(), "; abelian:", Q.is_abelian(), "; cyclic:", Q.is_cyclic())
print("element orders in D_4/Z:", sorted(q.order() for q in Q))
print("isomorphic to the Klein four-group:", Q.is_isomorphic(KleinFourGroup()))
