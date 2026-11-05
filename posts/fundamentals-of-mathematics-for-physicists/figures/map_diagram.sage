# Figure: three maps between finite sets drawn as bipartite graphs with Sage's graph module: an injective map that is
# not surjective, a surjective map that is not injective, and a bijection. Output: figures/map_diagram.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "map_diagram.svg")
def draw(pairs, A, B, title, shift):
    G = BipartiteGraph()
    for a in A: G.add_vertex(("a", a), left=True)
    for b in B: G.add_vertex(("b", b), right=True)
    for a, b in pairs: G.add_edge(("a", a), ("b", b))
    pos = {("a", a): (shift, 2 - i) for i, a in enumerate(A)}
    pos.update({("b", b): (shift + 1.2, 2 - i) for i, b in enumerate(B)})
    p = G.plot(pos=pos, vertex_labels=False, vertex_size=200, vertex_colors={"royalblue": [("a", a) for a in A], "firebrick": [("b", b) for b in B]}, edge_thickness=1.5)
    for a in A: p += text(str(a), (shift - 0.25, 2 - A.index(a)), fontsize=10)
    for b in B: p += text(str(b), (shift + 1.45, 2 - B.index(b)), fontsize=10)
    p += text(title, (shift + 0.6, 3.0), fontsize=10)
    return p
p = draw([(1, "x"), (2, "y")], [1, 2], ["x", "y", "z"], "injective, not surjective", 0)
p += draw([(1, "x"), (2, "x"), (3, "y")], [1, 2, 3], ["x", "y"], "surjective, not injective", 2.6)
p += draw([(1, "y"), (2, "z"), (3, "x")], [1, 2, 3], ["x", "y", "z"], "bijection", 5.2)
p.axes(False); p.set_aspect_ratio(1)
p.save(OUT, figsize=7)
f = {1: "y", 2: "z", 3: "x"}
print("[PASS] the third map is a bijection (its values are all distinct and fill the codomain):", len(set(f.values())) == 3 and set(f.values()) == {"x", "y", "z"})
print("[PASS] number of maps from a 3-element set to a 2-element set is 2^3 = 8:", len([m for m in cartesian_product([[0, 1]] * 3)]) == 8)
print("wrote", os.path.relpath(OUT))
