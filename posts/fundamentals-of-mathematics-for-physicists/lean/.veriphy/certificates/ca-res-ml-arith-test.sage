# name: ca-res-ml-arith-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.TaylorSeries.ml_arith: (1/2π)(M/r^(n+1))(2πr) = M/r^n
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
cases = [({'M': '1', 'r': '1', 'n': '0'}, True), ({'M': '3', 'r': '2', 'n': '4'}, False), ({'M': '1/2', 'r': '-3', 'n': '1'}, False)]
instances = [oracle_instance(*V(**kw), lhs=X(1/(2*pi)*(A(kw,'M')/A(kw,'r')^(A(kw,'n')+1))*(2*pi*A(kw,'r'))), rhs=X(A(kw,'M')/A(kw,'r')^A(kw,'n')), edge=e) for kw, e in cases]
