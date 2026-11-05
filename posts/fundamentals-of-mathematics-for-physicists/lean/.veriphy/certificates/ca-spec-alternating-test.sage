# name: ca-spec-alternating-test
# technique: test fallback (Veriphy test_fallback recipe): kernel-checked instances
# goal: sage_test KaluzaKlein.ComplexAnalysis.ZetaFunction.alternating_converges file "ca-spec-alternating-test.sage"
# description: s = 1, 1/2, 2; Lean runs the alternating series test (antitone terms tending to 0).
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
P = ('refine Antitone.tendsto_alternating_series_of_tendsto_zero ?_ ?_\n'
     '· intro m n hmn\n'
     '  have hm : (0 : ℝ) < (m : ℝ) + 1 := by positivity\n'
     '  exact one_div_le_one_div_of_le (Real.rpow_pos_of_pos hm _) '
     '(Real.rpow_le_rpow hm.le (by exact_mod_cast Nat.add_le_add_right hmn 1) (by norm_num))\n'
     '· have h : Filter.Tendsto (fun n : ℕ => ((n : ℝ) + 1) ^ (S : ℝ)) Filter.atTop Filter.atTop :=\n'
     '    (tendsto_rpow_atTop (by norm_num)).comp (Filter.tendsto_atTop_add_const_right _ 1 tendsto_natCast_atTop_atTop)\n'
     '  refine h.inv_tendsto_atTop.congr fun n => ?_\n'
     '  simp [one_div]')
instances = [kernel_instance(s, '(by norm_num)', proof=P.replace('S', s), edge=(s == '1')) for s in ['1', '1 / 2', '2']]
