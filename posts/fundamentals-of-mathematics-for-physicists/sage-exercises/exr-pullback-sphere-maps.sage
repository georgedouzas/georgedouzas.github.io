# Exercise exr-pullback-sphere-maps: pullbacks of the round metric of S^2 by three maps of the chart, with
# SageManifolds' DiffMap.pullback: a rotation about the Z-axis and the antipodal map are isometries, a map that
# halves the polar angle is not.
M = Manifold(2, "S2", structure="Riemannian"); X = M.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi"); th, ph = X[:]
g = M.metric("g"); g[0, 0] = 1; g[1, 1] = sin(th) ** 2
c = var("c", domain="real")
maps = {"rotation (th, ph) -> (th, ph + c)": [th, ph + c],
        "antipodal (th, ph) -> (pi - th, ph + pi)": [pi - th, ph + pi],
        "halving (th, ph) -> (th/2, ph)": [th / 2, ph]}
for name, comps in maps.items():
    F = M.diff_map(M, comps, name="F")
    h = F.pullback(g)
    h.apply_map(lambda e: e.simplify_full().trig_reduce())
    print(f"{name}:  F^* g =", h.display(), "  isometry:", h == g)
