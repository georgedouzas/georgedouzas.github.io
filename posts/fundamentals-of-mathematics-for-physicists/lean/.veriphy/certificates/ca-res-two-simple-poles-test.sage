# name: ca-res-two-simple-poles-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.IsolatedSingularities.two_simple_poles: 1/((z - a)(z - b)) has simple poles at a and b
# description: Sage computes lim (z - a) f and lim (z - b) f, finite and non-zero, and compares with 1/(a - b) and 1/(b - a)
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

def inst(a, b, edge=False):
    f = 1/((z - a)*(z - b))
    la, lb = limit((z - a)*f, z=a), limit((z - b)*f, z=b)
    assert not bool(la == 0) and not bool(lb == 0)
    return oracle_instance(*V(a=str(a), b=str(b)), lhs=abs(la - 1/(a - b)) + abs(lb - 1/(b - a)), rhs=0, edge=edge)
instances = [inst(1, 2), inst(0, 1, True), inst(-1, 3), inst(I, -I)]
