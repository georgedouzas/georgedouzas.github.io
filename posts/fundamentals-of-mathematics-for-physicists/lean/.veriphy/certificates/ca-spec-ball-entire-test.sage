# name: ca-spec-ball-entire-test
# technique: test fallback (Veriphy test_fallback recipe): kernel-checked instances
# goal: sage_test KaluzaKlein.ComplexAnalysis.BetaFunction.ball_volume_entire file "ca-spec-ball-entire-test.sage"
# description: dimensions s = 0, 2, 3 + i; Lean composes const_cpow and the entire 1/Gamma.
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
P = ('have h1 : DifferentiableAt ℂ (fun s : ℂ => (Real.pi : ℂ) ^ (s / 2)) @S@ := '
     '(differentiableAt_id.div_const (2 : ℂ)).const_cpow (Or.inl (Complex.ofReal_ne_zero.mpr Real.pi_ne_zero))\n'
     'have h2 : DifferentiableAt ℂ (fun s : ℂ => (Complex.Gamma (s / 2 + 1))⁻¹) @S@ := '
     'DifferentiableAt.comp (g := fun s => (Complex.Gamma s)⁻¹) (f := fun s : ℂ => s / 2 + 1) @S@ '
     '(Complex.differentiable_one_div_Gamma (@S@ / 2 + 1)) ((differentiableAt_id.div_const (2 : ℂ)).add_const 1)\n'
     'exact h1.mul h2')
instances = [kernel_instance(s, proof=P.replace('@S@', '(' + s + ' : ℂ)'), edge=(s == '0')) for s in ['0', '2', '3 + Complex.I']]
