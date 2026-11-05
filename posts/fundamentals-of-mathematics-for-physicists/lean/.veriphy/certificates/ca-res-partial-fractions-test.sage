# name: ca-res-partial-fractions-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.LaurentSeries.partial_fractions: 1/((z - 1)(z - 2)) = 1/(z - 2) - 1/(z - 1)
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
cases = [({'z': '0'}, True), ({'z': '3'}, False), ({'z': '3 / 2 * Complex.I'}, False), ({'z': '-1 / 2'}, False)]
instances = [oracle_instance(*V(**kw), lhs=X(1/((A(kw,'z')-1)*(A(kw,'z')-2))), rhs=X(1/(A(kw,'z')-2) - 1/(A(kw,'z')-1)), edge=e) for kw, e in cases]
