# name: dg-cc2-lib
# technique: shared helpers for the oracle tests of the Differential geometry chapter, part cc2 (not a certificate)
# goal: loaded by the dg-cc2-*-test.sage scripts with exec(open(<path>).read()); it defines curvature(), eq(), rel()
# description: Christoffel symbols, Riemann, Ricci, scalar curvature, lowered tensor and sectional curvature of a metric
#   matrix in coordinates (pure Python syntax, no preparsing), and exact-or-interval comparison of two values.
from sage.all import (SR, var, diff, matrix, Integer, QQ, ZZ, pi, sin, cos, tan, exp, cosh, sinh, sqrt, log,
                      RealIntervalField, ComplexBallField, identity_matrix)

_RI = RealIntervalField(160)


def curvature(g, coords):
    """Curvature of the metric matrix `g` (symbolic, over `coords`): a dictionary of callables
    Gam(k,i,j), Riem(r,s,m,n), low(r,s,m,n), Ric(s,n), R(), K() (surfaces), sect(X, Y), einstein(i,j)."""
    n = len(coords)
    ginv = g.inverse().apply_map(lambda e: e.simplify_full())
    Gam = [[[sum(ginv[k, l] * (diff(g[l, j], coords[i]) + diff(g[l, i], coords[j]) - diff(g[i, j], coords[l]))
                 for l in range(n)) / Integer(2) for j in range(n)] for i in range(n)] for k in range(n)]
    Gam = [[[Gam[k][i][j].simplify_full() for j in range(n)] for i in range(n)] for k in range(n)]

    def Riem(r, s, m, nu):
        return (diff(Gam[r][nu][s], coords[m]) - diff(Gam[r][m][s], coords[nu])
                + sum(Gam[r][m][l] * Gam[l][nu][s] - Gam[r][nu][l] * Gam[l][m][s] for l in range(n))).simplify_full()

    def low(r, s, m, nu):
        return sum(g[r, l] * Riem(l, s, m, nu) for l in range(n)).simplify_full()

    def Ric(s, nu):
        return sum(Riem(r, s, r, nu) for r in range(n)).simplify_full()

    def R():
        return sum(ginv[s, nu] * Ric(s, nu) for s in range(n) for nu in range(n)).simplify_full()

    def K():
        return (low(0, 1, 0, 1) / g.det()).simplify_full()

    def sect(X, Y):
        num = sum(low(r, s, m, nu) * X[r] * Y[s] * X[m] * Y[nu]
                  for r in range(n) for s in range(n) for m in range(n) for nu in range(n))
        gXX = sum(g[i, j] * X[i] * X[j] for i in range(n) for j in range(n))
        gYY = sum(g[i, j] * Y[i] * Y[j] for i in range(n) for j in range(n))
        gXY = sum(g[i, j] * X[i] * Y[j] for i in range(n) for j in range(n))
        return (num / (gXX * gYY - gXY ** 2)).simplify_full()

    def einstein(i, j):
        return (Ric(i, j) - R() * g[i, j] / Integer(2)).simplify_full()

    return {'ginv': ginv, 'Gam': lambda k, i, j: Gam[k][i][j], 'Riem': Riem, 'low': low, 'Ric': Ric, 'R': R,
            'K': K, 'sect': sect, 'einstein': einstein}


def _ri(v):
    return _RI(SR(v))


def eq(values, L, R, edge=False):
    """An oracle instance `L = R` at the binder values: exact when Sage simplifies the difference to zero, else
    by 160-bit intervals of |L − R| against 0."""
    try:
        Ls, Rs = SR(L).simplify_full(), SR(R).simplify_full()
        if bool((Ls - Rs).simplify_full() == 0):
            return oracle_instance(*values, lhs=Ls, rhs=Rs, edge=edge)
    except Exception:
        pass
    d = abs(_ri(L) - _ri(R))
    return oracle_instance(*values, lhs=d, rhs=_RI(0), edge=edge)


def rel(values, L, R, relation, edge=False):
    """An oracle instance `L <relation> R` decided on 160-bit intervals."""
    return oracle_instance(*values, lhs=_ri(L), rhs=_ri(R), relation=relation, edge=edge)


def B(c):
    """A Boolean as the rational 1 or 0, to compare with 1 in an oracle instance."""
    return QQ(1) if bool(c) else QQ(0)
