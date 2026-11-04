# Exercise exr-su3-cartan: the root system of su(3) with Sage's RootSystem: simple roots, positive roots, the Cartan
# matrix, the Dynkin diagram, and the weights of the fundamental representation from the WeylCharacterRing.
R = RootSystem(["A", 2])
L = R.root_lattice()
print("simple roots:", list(L.simple_roots()))
print("positive roots:", list(L.positive_roots()), "; number of roots:", len(list(L.roots())))
print("Cartan matrix A_ij = <alpha_i, alpha_j^vee>:"); print(R.cartan_matrix())
print("Dynkin diagram:"); print(R.dynkin_diagram())
A2 = WeylCharacterRing("A2", style="coroots")
print("weights of the fundamental 3 (orthogonal coordinates e1, e2, e3 with sum 0):", sorted(A2(1, 0).weight_multiplicities().items(), key=str))
print("weights of the adjoint 8 with multiplicities:", sorted(A2(1, 1).weight_multiplicities().items(), key=str))
print("the adjoint has dimension 8 and the zero weight has multiplicity 2 = rank:", A2(1, 1).degree() == 8 and A2(1, 1).weight_multiplicities()[A2.space()((0, 0, 0))] == 2)
