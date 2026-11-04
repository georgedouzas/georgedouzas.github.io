# Exercise exr-connected-components: connectedness and components with Sage's graph module as a finite model of a
# space: the vertices are regions, the edges say which regions touch.
G = Graph({"A": ["B"], "B": ["C"], "C": [], "D": ["E"], "E": []})
print("connected:", G.is_connected(), "; components:", G.connected_components(sort=True))
H = graphs.CycleGraph(6)
H1 = H.copy(); H1.delete_vertex(0)
H2 = H.copy(); H2.delete_vertices([0, 3])
print("the 6-cycle (a discrete circle) is connected:", H.is_connected())
print("removing one vertex leaves it connected:", H1.is_connected())
print("removing two opposite vertices gives", H2.connected_components_number(), "components (like removing two points from a circle)")
