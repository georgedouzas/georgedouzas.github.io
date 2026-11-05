# name: cm-rewrite
# technique: identities, values and inequalities of exp, log, sqrt, sin, cos, cosh, sinh checked in Sage first, then
#   closed in Lean by the mathlib rewrites listed for the goal (transcendental_rewrite recipe, HINTS given per goal shape)
# goal: an identity (or a value, ≠, or a numeric inequality) between transcendental expressions of the classical-mechanics chapter
# description: Sage confirms the statement exactly (simplify_full, canonicalize_radical, trig and log simplifications,
#   under the problem's sign hypotheses; a numeric inequality in 200-bit interval arithmetic); the Lean proof applies the
#   rewrites listed in HINTS for the goal's shape.
HINTS = [
    # (a substring of the goal's Lean text, the tactic closing it)
    ('(l * cos q) ^ 2 + (l * sin q) ^ 2 = l ^ 2', 'linear_combination l ^ 2 * Real.cos_sq_add_sin_sq q'),
    ('sqrt (4 * pi ^ 2 * m * a ^ 3 / k) = 2 * pi * sqrt (m * a ^ 3 / k)',
     'rw [show 4 * Real.pi ^ 2 * m * a ^ 3 / k = (2 * Real.pi) ^ 2 * (m * a ^ 3 / k) by ring, Real.sqrt_mul (by positivity), Real.sqrt_sq (by positivity)]'),
    ('2 * pi * p / sqrt (k / (m * p)) = 2 * pi * sqrt (m * p ^ 3 / k)',
     'rw [show m * p ^ 3 / k = p ^ 2 * (k / (m * p))⁻¹ by field_simp; ring, Real.sqrt_mul (by positivity), Real.sqrt_sq hp.le, Real.sqrt_inv]\nring'),
    ('|(1 - 0.0167) / (1 + 0.0167) - 0.967| < 0.0005', 'rw [abs_lt]\nconstructor <;> norm_num'),
    ('sin (arccos (-1 / e) - pi / 2) = 1 / e',
     'rw [Real.sin_sub_pi_div_two, Real.cos_arccos (by rw [neg_div, neg_le_neg_iff]; exact (div_le_one (by linarith)).mpr he.le) '
     '(by rw [neg_div]; linarith [one_div_pos.mpr (show (0 : ℝ) < e by linarith)])]\nring'),
    ('17.8 < 75.3 ^ (2 / 3)',
     'rw [show ((2 : ℝ) / 3) = 2 * (3 : ℝ)⁻¹ by norm_num, Real.rpow_mul (by norm_num), Real.lt_rpow_inv_iff_of_pos (by norm_num) (by positivity) (by norm_num)]\n'
     'first | (norm_num [Real.rpow_ofNat]; done) | (norm_num [Real.rpow_two, Real.rpow_natCast]; done) | (rw [show (3 : ℝ) = ((3 : ℕ) : ℝ) by norm_num, show (2 : ℝ) = ((2 : ℕ) : ℝ) by norm_num, Real.rpow_natCast, Real.rpow_natCast]; norm_num)'),
    ('75.3 ^ (2 / 3) < 17.85',
     'rw [show ((2 : ℝ) / 3) = 2 * (3 : ℝ)⁻¹ by norm_num, Real.rpow_mul (by norm_num), Real.rpow_inv_lt_iff_of_pos (by positivity) (by norm_num) (by norm_num)]\n'
     'first | (norm_num [Real.rpow_ofNat]; done) | (norm_num [Real.rpow_two, Real.rpow_natCast]; done) | (rw [show (3 : ℝ) = ((3 : ℕ) : ℝ) by norm_num, show (2 : ℝ) = ((2 : ℕ) : ℝ) by norm_num, Real.rpow_natCast, Real.rpow_natCast]; norm_num)'),
    ('|17.8 * (1 - 0.967) - 0.59| < 0.005', 'rw [abs_lt]\nconstructor <;> norm_num'),
    ('|17.8 * (1 + 0.967) - 35| < 0.05', 'rw [abs_lt]\nconstructor <;> norm_num'),
    ('sin (2 * α) = 2 * sin α * cos α', 'exact Real.sin_two_mul α'),
    ('sqrt (2 * G * M / R) ^ 2 = 2 * G * M / R', 'rw [Real.sq_sqrt (by positivity)]'),
    ('deriv (fun x => x) x - deriv (fun y => -y) y = 2', 'simp only [deriv_id\'\', deriv_neg\'\']; norm_num'),
    ('deriv (fun x => -x) x - deriv (fun y => y) y = -2', 'simp only [deriv_id\'\', deriv_neg\'\']; norm_num'),
    ('2 * pi ≠ 0', 'positivity'),
    ('sqrt (2 * (E - k * x ^ 2 / 2) / m) = sqrt (k / m) * sqrt (2 * E / k - x ^ 2)',
     'rw [← Real.sqrt_mul (by positivity)]\ncongr 1\nfield_simp\nring'),
    ('A * sin (ω * -(pi / (2 * ω))) = -A',
     'rw [show ω * -(Real.pi / (2 * ω)) = -(Real.pi / 2) by field_simp; ring, Real.sin_neg, Real.sin_pi_div_two]\nring'),
    ('A * sin (ω * (pi / (2 * ω))) = A', 'rw [show ω * (Real.pi / (2 * ω)) = Real.pi / 2 by field_simp; ring, Real.sin_pi_div_two]\nring'),
    ('2 * pi / sqrt (k / m) = 2 * pi * sqrt (m / k)',
     'rw [show m / k = (k / m)⁻¹ by rw [inv_div], Real.sqrt_inv]\nring'),
    ('|2 * 9.81 * 6371000 - 125000000| < 100000', 'rw [abs_lt]\nconstructor <;> norm_num'),
    ('11180 < sqrt (2 * 9.81 * 6371000)', 'rw [Real.lt_sqrt (by norm_num)]\nnorm_num'),
    ('sqrt (2 * 9.81 * 6371000) < 11181', 'rw [Real.sqrt_lt\' (by norm_num)]\nnorm_num'),
]
e = goal.expr
assert e is not None, 'the goal has no symbolic form; check the problem with print(goal)'
for h in problem.hyps:
    if h.expr is not None and h.rel in ('lt', 'le'):
        assume(h.expr < 0 if h.rel == 'lt' else h.expr <= 0)
def _forms(x):
    out = []
    for f in (lambda y: y.simplify_full(), lambda y: y.canonicalize_radical(),
              lambda y: y.simplify_log().simplify_full(), lambda y: y.simplify_trig().simplify_full(),
              lambda y: y.expand_trig().simplify_full(), lambda y: y.trig_expand(full=True).simplify_full(),
              lambda y: y.trig_reduce().simplify_full(), lambda y: y.expand().trig_reduce().simplify_full(), lambda y: y.expand_log().simplify_full().canonicalize_radical(),
              lambda y: y.rewrite('cosh(x)->exp(x)').rewrite('sinh(x)->exp(x)').simplify_full() if hasattr(y, 'rewrite') else y,
              lambda y: y.exp_simplify() if hasattr(y, 'exp_simplify') else y):
        try:
            out.append(f(SR(x)))
        except Exception:
            pass
    return out
forms = _forms(e)
if goal.rel == 'ne':
    assert all(bool(f != 0) for f in forms[:1]), f'Sage cannot confirm the inequation: lhs - rhs = {forms[0]}'
elif goal.rel in ('lt', 'le'):
    # an inequality between constants: lhs - rhs evaluated in interval arithmetic (200 bits)
    value = RealIntervalField(200)(e)
    assert value < 0 if goal.rel == 'lt' else value <= 0, f'Sage refutes the inequality: lhs - rhs ∈ {value}'
else:
    assert any(bool(f == 0) for f in forms), f'Sage cannot confirm the identity: lhs - rhs simplifies to {forms[0]}'
def _norm(t):
    for a, b in (('Real.', ''), ('rexp', 'exp'), ('√', 'sqrt '), ('π', 'pi'), (' ', '')):
        t = t.replace(a, b)
    return t
hints = [t for key, t in HINTS if _norm(key) in _norm(goal.text)]
assert hints, f'no rewrite is listed for {goal.text}; add one to HINTS'
proof = tactic(hints[0])
