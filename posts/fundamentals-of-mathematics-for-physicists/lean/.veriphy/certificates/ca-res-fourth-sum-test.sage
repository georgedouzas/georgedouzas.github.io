# name: ca-res-fourth-sum-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.ResidueTheorem.fourth_roots_sum: sum_(k<4) ζ i^k = 0
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
cases = [({'ζ': '0'}, True), ({'ζ': '1'}, False), ({'ζ': '(1 + Complex.I) / 2'}, False)]
instances = [oracle_instance(*V(**kw), lhs=X(sum(A(kw,'ζ')*I^k for k in range(4))), rhs=X(0), edge=e) for kw, e in cases]
