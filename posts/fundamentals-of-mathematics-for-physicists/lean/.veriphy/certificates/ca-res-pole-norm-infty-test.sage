# name: ca-res-pole-norm-infty-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.IsolatedSingularities.tendsto_norm_of_pole: |f| -> oo at a pole
# description: Sage computes the limit of the reciprocal |(z - c)^m/g| -> 0 along a ray into c
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

t = var('t')
cases = [(SR(1), 0, 1, 'fun _ => 1', True), (exp(z), 1, 2, 'Complex.exp', False), (z + 3, I, 3, 'fun w => w + 3', False)]
instances = [oracle_instance(*V(g=gt, c=str(c), m=str(m)), lhs=limit(abs(t^m / g.subs(z=c + t)), t=0, dir='+'), rhs=0, edge=e)
             for g, c, m, gt, e in cases]
