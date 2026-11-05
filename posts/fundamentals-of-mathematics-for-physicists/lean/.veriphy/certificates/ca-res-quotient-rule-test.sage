# name: ca-res-quotient-rule-test
# technique: test fallback (instances; oracle instances are evaluated by Sage exactly or in Arb's
#   rigorous ball arithmetic, kernel instances are checked by Lean)
# goal: KaluzaKlein.ComplexAnalysis.Residues.quotient_rule: the quotient rule Res g/h = g(z0)/(dh/dz)(z0)
# description: Sage computes the limit symbolically
SEED = 0
def V(*named, **kw):
    """The binders' values as Lean text, by binder name ('_' for an oracle instance's hypotheses)."""
    for d in named:
        kw.update(d)
    return [kw.get(name, '_') for name, _ in universal]
def X(v):
    """An exact symbolic value, radicals canonicalised (Sage's exact comparison then succeeds)."""
    return SR(v).canonicalize_radical().simplify_full()
n = var('n'); z = var('z'); x = var('x')

cases = [(SR(1), sin(z), 0, '0', 1, '1', 'fun _ => 1', 'Complex.sin', False),
         (exp(z), 1 + z^2, I, 'Complex.I', 2*I, '2 * Complex.I', 'Complex.exp', 'fun w => 1 + w ^ 2', False),
         (SR(1), 1 + z^4, exp(I*pi/4), 'Complex.exp (Real.pi / 4 * Complex.I)', 4*exp(3*I*pi/4), '4 * Complex.exp (3 * Real.pi / 4 * Complex.I)',
          'fun _ => 1', 'fun w => 1 + w ^ 4', False),
         (SR(0), z - 2, 2, '2', 1, '1', 'fun _ => 0', 'fun w => w - 2', True)]
instances = [oracle_instance(*V({'z₀': ct}, g=gt, h=ht, d=dt), lhs=limit((z - c) * g / h, z=c).simplify_full(), rhs=(g.subs(z=c) / d).simplify_full(),
             edge=e) for g, h, c, ct, d, dt, gt, ht, e in cases]
