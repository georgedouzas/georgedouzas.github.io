# Exercise exr-lrep-dsum-cartan: a direct sum of Lie algebras has a block-diagonal Cartan matrix and a disconnected Dynkin
# diagram: A1 x A1 = D2 (so(4)), A2 x A1 (su(3) + su(2)), and the irreducible representations as tensor products.
for ct in ["A1xA1", ["D", 2], "A2xA1"]:
    T = CartanType(ct)
    print(T, "Cartan matrix:"); print(CartanMatrix(T))
    print("   Dynkin diagram connected:", T.dynkin_diagram().to_undirected().is_connected())
R = WeylCharacterRing("A2xA1", style="coroots")
print("irreducible (3, 2) of su(3) + su(2) has dimension", R(1, 0, 1).degree(), "= 3 * 2; (8, 3):", R(1, 1, 2).degree())
