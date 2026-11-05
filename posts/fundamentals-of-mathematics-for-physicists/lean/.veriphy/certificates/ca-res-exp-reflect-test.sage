# name: ca-res-exp-reflect-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.Residues.exp_reflect: e^(iθ) + e^(i(π - θ)) = 2i sin θ
# description: Sage evaluates both sides exactly
SEED = 0
def V(*named, **kw):
    """The binders' values as Lean text, by binder name ('_' for an oracle instance's hypotheses)."""
    for d in named:
        kw.update(d)
    return [kw.get(name, '_') for name, _ in universal]
def X(v):
    """An exact symbolic value, radicals canonicalised (Sage's exact comparison then succeeds)."""
    return SR(v).canonicalize_radical().simplify_full()
RIq = RealIntervalField(128); CIq = ComplexIntervalField(128)
def G(a, b):
    return RIq(abs(CIq(a) - CIq(b)))
instances = [oracle_instance(*V(θ=tt), lhs=G(exp(I*t) + exp(I*(pi - t)), 2*sin(t)*I), rhs=RIq(0), edge=(tt == '0')) for tt, t in
             [('0', 0), ('Real.pi / 4', pi/4), ('Real.pi / 3', pi/3), ('Real.pi / 2', pi/2)]]
