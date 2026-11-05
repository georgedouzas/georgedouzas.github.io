# name: ca-res-one-add-sq-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.TaylorSeries.one_add_sq_factor: 1 + z^2 = (z - i)(z + i)
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
cases = [({'z': '0'}, True), ({'z': '1'}, False), ({'z': '2 + Complex.I'}, False), ({'z': '-3'}, False)]
instances = [oracle_instance(*V(**kw), lhs=X(1 + A(kw,'z')^2), rhs=X((A(kw,'z') - I)*(A(kw,'z') + I)), edge=e) for kw, e in cases]
