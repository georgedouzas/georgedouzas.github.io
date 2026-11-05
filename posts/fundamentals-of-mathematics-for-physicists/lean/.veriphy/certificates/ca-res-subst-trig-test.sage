# name: ca-res-subst-trig-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.ResidueWorked.subst_trig: cos θ = (z + 1/z)/2, sin θ = (z - 1/z)/(2i) for z = e^(iθ)
# description: Sage evaluates both identities exactly (the derivative is checked by Lean in the proof)
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
def d(t):
    u = exp(I*t)
    return G(cos(t), (u + 1/u)/2) + G(sin(t), (u - 1/u)/(2*I))
instances = [oracle_instance(*V(θ=tt), lhs=d(t), rhs=RIq(0), edge=(tt == '0')) for tt, t in
             [('0', 0), ('Real.pi / 3', pi/3), ('Real.pi / 4', pi/4), ('2', 2)]]
