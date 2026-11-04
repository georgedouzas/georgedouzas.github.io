# Exercise exr-vcp-generations-weyl: for a compact coset G/H with rank G = rank H the Euler number is |W_G|/|W_H|. Checks:
# S^2 = SU(2)/U(1) (2), CP^2 = SU(3)/U(2) (3), the three nearly-Kahler cosets G_2/SU(3) (2), Sp(4)/(SU(2) x U(1)) (4),
# SU(3)/U(1)^2 (6), and the six-sphere G_2/SU(3) = S^6 (chi(S^6) = 2).
W = lambda ct: WeylGroup(ct).order() if ct else 1
cases = [("S^2 = SU(2)/U(1)", ["A", 1], None), ("CP^2 = SU(3)/U(2)", ["A", 2], ["A", 1]),
         ("G_2/SU(3) = S^6", ["G", 2], ["A", 2]), ("Sp(4)/(SU(2) x U(1))", ["C", 2], ["A", 1]), ("SU(3)/(U(1) x U(1))", ["A", 2], None)]
for name, G, H in cases:
    print(name, ": |W_G| =", W(G), ", |W_H| =", W(H), ", chi =", W(G) // W(H))
print("nearly-Kahler cosets: chi/2 =", [W(G) // W(H) // 2 for name, G, H in cases[2:]], "generations")
print("SU(2) x SU(2) (rank 2) has no torus quotient of equal rank here; as a group manifold its Euler number is 0 (it has a nowhere-vanishing vector field)")
