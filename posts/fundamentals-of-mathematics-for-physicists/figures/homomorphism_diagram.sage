# Figure: a homomorphism and its fibres. The map Z_6 -> Z_2, k -> k mod 2, drawn as a bipartite graph with Sage's
# graph module: the fibre over 0 is the kernel {0, 2, 4} and the fibre over 1 is the other coset {1, 3, 5}; the
# fibres are the cosets of the kernel, which is the first isomorphism theorem in a picture.
# Output: figures/homomorphism_diagram.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "homomorphism_diagram.svg")
G = BipartiteGraph()
for k in range(6): G.add_vertex(("a", k), left=True)
for j in range(2): G.add_vertex(("b", j), right=True)
for k in range(6): G.add_edge(("a", k), ("b", k % 2))
pos = {("a", k): (0, 2.5 - k) for k in range(6)}
pos.update({("b", j): (2.0, 1.0 - 2 * j) for j in range(2)})
p = G.plot(pos=pos, vertex_labels=False, vertex_size=260, vertex_colors={"royalblue": [("a", k) for k in range(0, 6, 2)], "lightsalmon": [("a", k) for k in range(1, 6, 2)], "lightgrey": [("b", 0), ("b", 1)]}, edge_colors={"royalblue": [(("a", k), ("b", 0), None) for k in range(0, 6, 2)], "lightsalmon": [(("a", k), ("b", 1), None) for k in range(1, 6, 2)]}, edge_thickness=1.5)
for k in range(6): p += text(str(k), (-0.35, 2.5 - k), fontsize=10)
for j in range(2): p += text(str(j), (2.35, 1.0 - 2 * j), fontsize=10)
p += text(r"$\mathbb{Z}_6$", (0, 3.4), fontsize=11) + text(r"$\mathbb{Z}_2$", (2.0, 3.4), fontsize=11) + text(r"$k\mapsto k\ \mathrm{mod}\ 2$", (1.0, 3.9), fontsize=10)
p += text("kernel $\\{0,2,4\\}$", (-1.2, 2.5), fontsize=9, color="royalblue") + text("coset $\\{1,3,5\\}$", (-1.2, -2.5), fontsize=9, color="darkorange")
p.axes(False); p.set_aspect_ratio(1)
p.save(OUT, figsize=4.2)
Z6, Z2 = Zmod(6), Zmod(2)
f = lambda k: Z2(ZZ(k) % 2)
print("[PASS] k -> k mod 2 is a homomorphism Z_6 -> Z_2:", all(f(a + b) == f(a) + f(b) for a in Z6 for b in Z6))
print("[PASS] its kernel is {0, 2, 4} and the fibre over 1 is {1, 3, 5}:", sorted(ZZ(k) for k in Z6 if f(k) == 0) == [0, 2, 4] and sorted(ZZ(k) for k in Z6 if f(k) == 1) == [1, 3, 5])
print("wrote", os.path.relpath(OUT))
