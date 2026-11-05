# Figure: finite models of connectedness with Sage's graph module. Left: a graph with two connected components.
# Middle: the six-cycle, a discrete circle, stays connected after one vertex is removed. Right: removing two opposite
# vertices splits it into two components, as removing two points splits a circle into two arcs.
# Output: figures/components_graph.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "components_graph.svg")
def shifted(G, pos, dx, title, colours):
    pos = {v: (x + dx, y) for v, (x, y) in pos.items()}
    p = G.plot(pos=pos, vertex_size=250, vertex_colors=colours, edge_thickness=1.5, vertex_labels=True)
    p += text(title, (dx + 1.0, 1.9), fontsize=10)
    return p
G = Graph({"A": ["B"], "B": ["C"], "D": ["E"]})
posG = {"A": (0, 0.8), "B": (1, 1.2), "C": (2, 0.8), "D": (0.5, -0.4), "E": (1.5, -0.4)}
comps = G.connected_components(sort=True)
p = shifted(G, posG, 0, "two components", {"royalblue": comps[0], "firebrick": comps[1]})
H = graphs.CycleGraph(6)
posH = {i: (1 + cos(pi / 3 * i), 0.4 + sin(pi / 3 * i)) for i in range(6)}
H1 = H.copy(); H1.delete_vertex(0)
p += shifted(H1, posH, 3.5, "one vertex removed: connected", {"royalblue": list(H1.vertices())})
H2 = H.copy(); H2.delete_vertices([0, 3])
c2 = H2.connected_components(sort=True)
p += shifted(H2, posH, 7.0, "two removed: two components", {"royalblue": c2[0], "firebrick": c2[1]})
p.axes(False); p.set_aspect_ratio(1)
p.save(OUT, figsize=8)
print("[PASS] the first graph has two components:", G.connected_components_number() == 2)
print("[PASS] the six-cycle minus one vertex is connected and minus two opposite vertices has two components:", H1.is_connected() and H2.connected_components_number() == 2)
print("wrote", os.path.relpath(OUT))
