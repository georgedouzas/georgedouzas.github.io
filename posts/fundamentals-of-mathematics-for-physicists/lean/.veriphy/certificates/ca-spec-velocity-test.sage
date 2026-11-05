# name: ca-spec-velocity-test
# technique: test fallback (Veriphy test_fallback recipe): kernel-checked instances
# goal: sage_test KaluzaKlein.ComplexAnalysis.WickRotation.velocity_rotation file "ca-spec-velocity-test.sage"
# description: paths x = id, exp, z^2 at t = 0, 1, i; Lean composes the chain rule.
SEED = 0

# shared oracle helpers: exact comparison when Sage decides it, else rigorous balls (160 bits)
from sage.all import (SR, ZZ, QQ, I, pi, oo, sqrt, exp, log, sin, cos, gamma, zeta, psi, factorial, bernoulli,
                      harmonic_number, euler_gamma, integrate, limit, var, assume, forget, diff, conjugate, beta,
                      ComplexBallField, RealIntervalField, real_part, imag_part)
from sage.all import sum as ssum
_C = ComplexBallField(160)
_R = RealIntervalField(160)
from sage.rings.complex_arb import ComplexBall as _CB
from sage.rings.real_arb import RealBall as _RB
def _ball(v):
    if isinstance(v, (_CB, _RB)):
        return _C(v)
    return _C(SR(v))
def babs(v):
    """|v| as a rigorous ball (Sage's symbolic abs may drop the absolute value of a complex power)"""
    return _ball(v).abs()
def _ri(rb):
    return _R(rb.lower(), rb.upper())
def eq(values, L, R, edge=False):
    """the claim's two sides at the values: exact when Sage simplifies the difference to 0, else |L - R| in balls"""
    try:
        if isinstance(L, (_CB, _RB)) or isinstance(R, (_CB, _RB)):
            raise ValueError('balls')
        Ls, Rs = SR(L).simplify_full(), SR(R).simplify_full()
        if bool(Ls - Rs == 0):
            return oracle_instance(*values, lhs=Ls, rhs=Rs, edge=edge)
    except Exception:
        pass
    d = (_ball(L) - _ball(R)).abs()
    return oracle_instance(*values, lhs=_ri(d), rhs=_R(0), edge=edge)
def rel(values, L, R, relation, edge=False):
    """a real inequality between the sides, decided on rigorous intervals"""
    return oracle_instance(*values, lhs=_ri(_ball(L).real()), rhs=_ri(_ball(R).real()), relation=relation, edge=edge)
def P(x, hx, t):
    return ('have h := (' + hx + ' (Complex.I * ' + t + ')).hasDerivAt.comp ' + t + ' ((hasDerivAt_id ' + t + ').const_mul Complex.I)\n'
            'have e : Complex.I * deriv (' + x + ') (Complex.I * ' + t + ') = deriv (' + x + ') (Complex.I * ' + t + ') * (Complex.I * 1) := by ring\n'
            'rw [e]\nexact h')
H2 = '(by fun_prop : Differentiable ℂ (fun z : ℂ => z ^ 2))'
instances = [kernel_instance('id', 'differentiable_id', '0', proof=P('id', 'differentiable_id', '(0 : ℂ)'), edge=True),
             kernel_instance('Complex.exp', 'Complex.differentiable_exp', '1', proof=P('Complex.exp', 'Complex.differentiable_exp', '(1 : ℂ)')),
             kernel_instance('fun z : ℂ => z ^ 2', H2, 'Complex.I', proof=P('fun z : ℂ => z ^ 2', H2, 'Complex.I'))]
