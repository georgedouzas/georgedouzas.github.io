# Shared helper for the geometry exercises, built on SageManifolds: a manifold with one chart and a metric given by
# its component functions; Christoffel symbols, Riemann and Ricci tensors and the scalar curvature come from the
# metric's own methods (the same sign conventions as the conventions chapter: R^r_{s m n} = d_m Gamma^r_{n s} - ...,
# which is SageManifolds' convention).
from sage.all import *

def chart_metric(names, components, structure='Riemannian', ranges=None):
    """Return (M, X, g) for the chart with coordinate `names` ('th ph') and the metric matrix `components(X)`.

    `components` maps the chart coordinates (a tuple of symbolic variables) to a list of lists; `structure` is
    'Riemannian' or 'Lorentzian'; `ranges` optionally restricts coordinates, e.g. 'th:(0,pi) ph:(0,2*pi)'."""
    n = len(names.split())
    M = Manifold(n, 'M', structure=structure)
    X = M.chart(ranges if ranges else names)
    g = M.metric('g')
    comps = components(X[:])
    for i in range(n):
        for j in range(n):
            if comps[i][j] != 0:
                g[i, j] = comps[i][j]
    return M, X, g

def christoffel(g):
    """The non-zero Christoffel symbols Gamma^k_ij (i <= j) as a dict."""
    G = g.christoffel_symbols()
    n = g.domain().dim(); X = g.domain().default_chart()
    out = {}
    for k in range(n):
        for i in range(n):
            for j in range(i, n):
                v = G[k, i, j].expr().simplify_full()
                if v != 0: out[(str(X[k]), str(X[i]), str(X[j]))] = v
    return out

def curvature(g):
    """(Riemann tensor, Ricci tensor, scalar curvature) of the metric, as SageManifolds tensor fields and a scalar."""
    return g.riemann(), g.ricci(), g.ricci_scalar()
