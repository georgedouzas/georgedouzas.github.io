# name: ca-res-casorati-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.IsolatedSingularities.casorati: e^(1/z) = w has infinitely many solutions in 0 < |z| < ε
# description: Sage verifies exactly that z_k = 1/(Log w + 2πik), k = K..K+39, solve e^(1/z) = w with |z_k| < ε (40 distinct solutions)
SEED = 0
def V(*named, **kw):
    """The binders' values as Lean text, by binder name ('_' for an oracle instance's hypotheses)."""
    for d in named:
        kw.update(d)
    return [kw.get(name, '_') for name, _ in universal]
def X(v):
    """An exact symbolic value, radicals canonicalised (Sage's exact comparison then succeeds)."""
    return SR(v).canonicalize_radical().simplify_full()
def count(w, eps):
    K = ceil(1/eps/(2*pi).n() + 2)
    sols = [1/(log(w) + 2*pi*I*k) for k in range(K, K + 40)]
    ok = [s for s in sols if bool((exp(1/s) - w).simplify_full() == 0) and bool(abs(s) < eps)]
    return len(set(ok))
instances = [oracle_instance(*V(w=wt, ε=et), lhs=count(w, eps), rhs=40, edge=e) for wt, w, et, eps, e in
             [('1', 1, '1', 1, True), ('-1', -1, '1 / 10', 1/10, False), ('Complex.I', I, '1 / 100', 1/100, False), ('2', 2, '1 / 2', 1/2, False)]]
