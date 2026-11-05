# name: ca-res-pole-iff-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.IsolatedSingularities.isPoleOfOrder_iff: a pole of order m: (z - c)^m f analytic and non-zero at c
# description: Sage checks, for poles of known order, that lim (z - c)^m f is finite and non-zero (both sides of the equivalence hold)
SEED = 0
def V(*named, **kw):
    """The binders' values as Lean text, by binder name ('_' for an oracle instance's hypotheses)."""
    for d in named:
        kw.update(d)
    return [kw.get(name, '_') for name, _ in universal]
def X(v):
    """An exact symbolic value, radicals canonicalised (Sage's exact comparison then succeeds)."""
    return SR(v).canonicalize_radical().simplify_full()
n = var('n'); z = var('z'); x = var('x')

cases = [(1/z, 0, 1, 'fun w => 1 / w', True), (1/((z - 1)*(z - 2)), 1, 1, 'fun w => 1 / ((w - 1) * (w - 2))', False),
         (sin(z)/z^3, 0, 2, 'fun w => Complex.sin w / w ^ 3', False), (1/(z^2 + 1)^3, I, 3, 'fun w => 1 / (w ^ 2 + 1) ^ 3', False)]
def ok(f, c, m):
    L = limit((z - c)^m * f, z=c)
    return 1 if (L not in [oo, -oo, unsigned_infinity] and not bool(L == 0)) else 0
instances = [oracle_instance(*V(f=ft, c=str(c), m=str(m)), lhs=ok(f, c, m), rhs=1, edge=e) for f, c, m, ft, e in cases]
