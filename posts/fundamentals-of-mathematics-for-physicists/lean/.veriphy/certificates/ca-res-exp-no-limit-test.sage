# name: ca-res-exp-no-limit-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.IsolatedSingularities.exp_inv_no_limit: e^(a/z) has no limit at 0
# description: Sage computes the two one-sided limits along the ray z = a x (e^(1/x)): +oo from the right, 0 from the left, which differ
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

instances = [oracle_instance(*V(a=str(a)), lhs=limit(exp(a/(a*x)), x=0, dir='+'), rhs=limit(exp(a/(a*x)), x=0, dir='-'), relation='>',
             edge=(a == 1)) for a in [1, 2, I, -1]]
