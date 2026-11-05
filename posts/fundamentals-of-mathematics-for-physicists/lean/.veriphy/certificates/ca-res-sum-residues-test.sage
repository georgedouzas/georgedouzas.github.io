# name: ca-res-sum-residues-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.ResidueTheorem.sum_residues_zero: the finite residues and the residue at infinity sum to 0
# description: Arb's rigorous circle integrals for explicit principal parts
SEED = 0
def V(*named, **kw):
    """The binders' values as Lean text, by binder name ('_' for an oracle instance's hypotheses)."""
    for d in named:
        kw.update(d)
    return [kw.get(name, '_') for name, _ in universal]
def X(v):
    """An exact symbolic value, radicals canonicalised (Sage's exact comparison then succeeds)."""
    return SR(v).canonicalize_radical().simplify_full()
CBF = ComplexBallField(128)
RIFp = RealIntervalField(128)
CI = CBF(0, 1)
TWO_PI_I = 2 * CBF.pi() * CI
def circ(f, c, rho):
    """Arb's rigorous enclosure of the circle integral of f over |z - c| = rho."""
    c = CBF(c); rho = CBF(rho)
    return CBF.integral(lambda t, _: f(c + rho * (CI * t).exp()) * CI * rho * (CI * t).exp(), 0, 2 * CBF.pi().real())
def seg(f, a, b):
    """Arb's rigorous enclosure of the integral of f(t) dt over [a, b]."""
    return CBF.integral(lambda t, _: f(t), CBF(a), CBF(b))
def gap(x, y):
    """The interval of |x - y| (contains 0 exactly when the enclosures are compatible)."""
    d = (CBF(x) - CBF(y)).abs()
    return RIFp(d.lower(), d.upper())
def bnd(x):
    """The interval of |x|."""
    d = CBF(x).abs()
    return RIFp(d.lower(), d.upper())
def eq(x, y, *values, edge=False):
    return oracle_instance(*values, lhs=gap(x, y), rhs=RIFp(0), relation='==', edge=edge)

def inst(terms, g, gt, rho, edge=False):
    F = lambda w: sum(CBF(c)/(w - CBF(p))^(k + 1) for c, p, k in terms) + g(w)
    res = sum(CBF(c) for c, p, k in terms if k == 0)
    st = 'Finset.range %d' % len(terms)
    wt = 'fun i => ' + ' '.join('if i = %d then (%s : ℂ) else' % (i, str(p)) for i, (c, p, k) in enumerate(terms)) + ' 0'
    ct = 'fun i => ' + ' '.join('if i = %d then (%s : ℂ) else' % (i, str(c)) for i, (c, p, k) in enumerate(terms)) + ' 0'
    kt = 'fun i => ' + ' '.join('if i = %d then %d else' % (i, k) for i, (c, p, k) in enumerate(terms)) + ' 0'
    return eq(res - circ(F, 0, rho) / TWO_PI_I, 0, *V(ι='ℕ', s=st, ρ=str(rho), w=wt, c=ct, k=kt, g=gt), edge=edge)
instances = [inst([], lambda w: CBF(0), 'fun _ => 0', 1, True), inst([(-1, 1, 0), (1, 2, 0)], lambda w: CBF(0), 'fun _ => 0', 3),
             inst([(1, 0, 0), (0, 1, 0), (2, 1, 1)], lambda w: w^2, 'fun w => w ^ 2', 2)]
