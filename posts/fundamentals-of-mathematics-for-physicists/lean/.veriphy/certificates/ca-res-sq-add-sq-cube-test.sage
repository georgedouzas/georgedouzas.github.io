# name: ca-res-sq-add-sq-cube-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.IsolatedSingularities.sq_add_sq_cube: (z^2 + b^2)^3 = (z - ib)^3 (z + ib)^3
# description: Sage evaluates both sides exactly at the instance
SEED = 0
def V(*named, **kw):
    """The binders' values as Lean text, by binder name ('_' for an oracle instance's hypotheses)."""
    for d in named:
        kw.update(d)
    return [kw.get(name, '_') for name, _ in universal]
def X(v):
    """An exact symbolic value, radicals canonicalised (Sage's exact comparison then succeeds)."""
    return SR(v).canonicalize_radical().simplify_full()
def A(kw, k):
    return SR(sage_eval(kw[k].replace('Complex.I', 'I'), locals={'I': I}))
z = var('z')
cases = [({'z': '0', 'b': '0'}, True), ({'z': '1', 'b': '1'}, False), ({'z': '2 + Complex.I', 'b': '3'}, False)]
instances = [oracle_instance(*V(**kw), lhs=X((A(kw,'z')^2 + A(kw,'b')^2)^3), rhs=X((A(kw,'z') - I*A(kw,'b'))^3*(A(kw,'z') + I*A(kw,'b'))^3), edge=e) for kw, e in cases]
