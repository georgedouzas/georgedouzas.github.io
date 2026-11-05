# name: ca-res-partial-ab-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.Residues.partial_fraction_ab: partial fractions of 1/((z - a)(z - b))
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
cases = [({'a': '1', 'b': '2', 'z': '0'}, True), ({'a': '1', 'b': '2', 'z': '3 * Complex.I'}, False), ({'a': 'Complex.I', 'b': '-Complex.I', 'z': '2'}, False)]
instances = [oracle_instance(*V(**kw), lhs=X(1/((A(kw,'z')-A(kw,'a'))*(A(kw,'z')-A(kw,'b')))), rhs=X(1/(A(kw,'a')-A(kw,'b'))/(A(kw,'z')-A(kw,'a')) + 1/(A(kw,'b')-A(kw,'a'))/(A(kw,'z')-A(kw,'b'))), edge=e) for kw, e in cases]
