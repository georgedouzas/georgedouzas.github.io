# Exercise exr-branching-levi: branching rules with Sage's WeylCharacterRing: the fundamental and the adjoint of su(3)
# restricted to the su(2) of the first two rows and columns (the Levi subalgebra A1), and to the maximal torus.
A2 = WeylCharacterRing("A2", style="coroots"); A1 = WeylCharacterRing("A1", style="coroots")
for name, rep in [("3", A2(1, 0)), ("3bar", A2(0, 1)), ("8", A2(1, 1))]:
    print(f"{name} -> su(2):", rep.branch(A1, rule="levi"))
ad = A2(1, 1)
print("under the maximal torus u(1) x u(1) every weight is a one-dimensional representation; the adjoint has", len(ad.weight_multiplicities()), "distinct weights, the zero weight twice")
print("dimension check: 8 =", sum(c * A1((k,)).degree() for k, c in [(0, 1), (1, 2), (2, 1)]), "(singlet + two doublets + triplet)")
print("3 (x) 3bar = 8 + 1:", A2(1, 0) * A2(0, 1) == A2(1, 1) + A2(0, 0))
