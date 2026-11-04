# Exercise exr-cff-hodge-table: with the standard embedding the 27s are counted by h^{2,1}, the 27bars by h^{1,1}; net
# families |h11 - h21| = |chi|/2, vector-like pairs min(h11, h21).
data = [("quintic", 1, 101), ("resolved T^6/Z_3", 36, 0), ("Y/G (Braun-Candelas-Davies)", 1, 4), ("mirror of Y/G", 4, 1), ("(5, 8)", 5, 8), ("Yau (6, 9)", 6, 9), ("(2, 2)", 2, 2)]
print("%-30s %5s %6s %8s %5s %7s" % ("manifold", "n_27", "n_27b", "families", "chi", "pairs"))
for name, h11, h21 in data:
    print("%-30s %5d %6d %8d %5d %7d" % (name, h21, h11, abs(h11 - h21), 2 * (h11 - h21), min(h11, h21)))
