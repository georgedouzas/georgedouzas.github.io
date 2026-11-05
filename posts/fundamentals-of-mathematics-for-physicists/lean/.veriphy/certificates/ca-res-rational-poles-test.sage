# name: ca-res-rational-poles-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.IsolatedSingularities.rational_poles: (z^2 + 1)/(z (z - a)^2): a simple pole at 0 and a double pole at a
# description: Sage computes lim z f and lim (z - a)^2 f, finite and non-zero, and compares with 1/a^2 and (a^2 + 1)/a
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

def inst(a, edge=False):
    f = (z^2 + 1)/(z*(z - a)^2)
    l0, la = limit(z*f, z=0), limit((z - a)^2*f, z=a)
    assert not bool(l0 == 0) and not bool(la == 0)
    return oracle_instance(*V(a=str(a)), lhs=abs(l0 - 1/a^2) + abs(la - (a^2 + 1)/a), rhs=0, edge=edge)
instances = [inst(1, True), inst(2), inst(-1/2)]
