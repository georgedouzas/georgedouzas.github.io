# name: ca-spec-identity-test
# technique: test fallback (Veriphy test_fallback recipe): kernel-checked instances
# goal: sage_test KaluzaKlein.ComplexAnalysis.AnalyticContinuation.identity_theorem file "ca-spec-identity-test.sage"
# description: pairs f, g that agree near z0 on a connected open set; Lean checks EqOn by computation (rfl, ring, simp).
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
instances = [
  kernel_instance('Complex.exp', 'Complex.exp', 'Set.univ', 'isOpen_univ', 'isPreconnected_univ',
    'Complex.differentiable_exp.differentiableOn', 'Complex.differentiable_exp.differentiableOn', '0', 'Set.mem_univ 0',
    'Filter.Frequently.of_forall (fun _ => rfl)', proof='exact fun _ _ => rfl', edge=True),
  kernel_instance('fun z : ℂ => (z + 1) ^ 2', 'fun z : ℂ => z ^ 2 + 2 * z + 1', 'Metric.ball (0 : ℂ) 1', 'Metric.isOpen_ball',
    '(convex_ball (0 : ℂ) 1).isPreconnected', 'by fun_prop', 'by fun_prop', '0', 'Metric.mem_ball_self one_pos',
    'Filter.Frequently.of_forall (fun z => by beta_reduce; ring)', proof='intro z _\nbeta_reduce\nring'),
  kernel_instance('Complex.cos', 'fun z : ℂ => Complex.cos (-z)', 'Set.univ', 'isOpen_univ', 'isPreconnected_univ',
    'Complex.differentiable_cos.differentiableOn', 'by fun_prop', '0', 'Set.mem_univ 0',
    'Filter.Frequently.of_forall (fun z => by simp)', proof='intro z _\nsimp'),
]
