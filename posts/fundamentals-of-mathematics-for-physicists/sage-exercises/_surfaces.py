# Shared helper for the exercises on surfaces in R^3: for a parametrisation E(u, v) return the induced metric g, the
# unit normal n = E_u x E_v / |E_u x E_v|, the second fundamental form h_ij = n . d_i d_j E, the shape operator
# S = g^{-1} h and the Gauss curvature det h / det g (plain symbolic vectors and matrices).
from sage.all import *

def second_form(E, u, v):
    E = vector(SR, E)
    Eu, Ev = E.diff(u), E.diff(v)
    cr = Eu.cross_product(Ev)
    n = cr/sqrt(cr.dot_product(cr)).simplify_full()
    g = matrix(SR, [[Eu.dot_product(Eu), Eu.dot_product(Ev)], [Ev.dot_product(Eu), Ev.dot_product(Ev)]]).simplify_full()
    h = matrix(SR, [[n.dot_product(E.diff(a).diff(b)) for b in (u, v)] for a in (u, v)]).simplify_full()
    S = (g.inverse()*h).simplify_full()
    K = (h.det()/g.det()).simplify_full()
    return g, n, h, S, K
