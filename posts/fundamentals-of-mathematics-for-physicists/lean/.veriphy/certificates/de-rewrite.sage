# name: de-rewrite
# technique: identities and values of exp, log, sqrt, sin, cos checked in Sage first, then closed in Lean by the named
#   mathlib rewrites listed for the goal (transcendental_rewrite recipe, REWRITES given per goal shape)
# goal: an identity (or a value, or ≠) between exp, log, sqrt, sin, cos expressions of the differential-equations chapter
# description: Sage confirms the statement exactly (simplify_full, canonicalize_radical, trig and log simplifications,
#   under the problem's sign hypotheses); the Lean proof applies the rewrites listed in HINTS for the goal's shape.
HINTS = [
    # (a substring of the goal's Lean text, the tactic closing it)
    ('pi / γ / (2 * pi / ωd) = ωd / (2 * γ)', 'have hπ := Real.pi_pos.ne\'\nfirst | (field_simp; done) | (field_simp; ring)'),
    ('F / (2 * γ * ω₀) = ω₀ / (2 * γ) * F / ω₀ ^ 2', 'first | (field_simp; done) | (field_simp; ring)'),
    ('2 * γ * ω * F / ((ω₀ ^ 2 - ω ^ 2) ^ 2 + 4 * γ ^ 2 * ω ^ 2) /', 'rw [div_div_div_cancel_right₀ hΔ, div_eq_div_iff (mul_ne_zero hF hd) hd]\nring'),
    ('(ω₀ ^ 2 - ω ^ 2) * (F * (ω₀ ^ 2 - ω ^ 2) / ((ω₀ ^ 2 - ω ^ 2) ^ 2 + 4 * γ ^ 2 * ω ^ 2)) +', 'rw [mul_div_assoc\', mul_div_assoc\', ← add_div, div_eq_iff hΔ]\nring'),
    ('-(2 * γ * ω) * (F * (ω₀ ^ 2 - ω ^ 2) /', 'rw [mul_div_assoc\', mul_div_assoc\', ← add_div, div_eq_iff hΔ]\nring'),
    ('(F * (ω₀ ^ 2 - ω ^ 2) / ((ω₀ ^ 2 - ω ^ 2) ^ 2 + 4 * γ ^ 2 * ω ^ 2)) ^ 2 +', 'rw [div_pow, div_pow, ← add_div, div_eq_div_iff (pow_ne_zero 2 hΔ) hΔ]\nring'),
    ('sin x * -cos (1 - x) - cos x * sin (1 - x) = -sin 1',
     'have h := Real.sin_add x (1 - x)\nrw [show x + (1 - x) = 1 by ring] at h\nlinarith'),
    ("(ua * ub' - ua' * ub) / -(p * (ua * ub' - ua' * ub)) = -1 / p",
     "rw [div_eq_div_iff (neg_ne_zero.mpr hW) hp]\nring"),
    ('y₀ / (1 - y₀ * 0) = y₀', 'simp'),
    ('1 / (1 + 0 * exp (-x)) = 1', 'simp'),
    ('187 / 200 < sqrt (7 / 8)', 'rw [Real.lt_sqrt (by norm_num)]\nnorm_num'),
    ('sqrt (7 / 8) < 117 / 125', "rw [Real.sqrt_lt' (by norm_num)]\nnorm_num"),
    ('413 / 200 < 8 / sqrt 15',
     'have h := Real.sqrt_pos.mpr (show (0 : ℝ) < 15 by norm_num)\nhave h2 := Real.sq_sqrt (show (0 : ℝ) ≤ 15 by norm_num)\n'
     'rw [lt_div_iff₀ h]\nnlinarith'),
    ('8 / sqrt 15 < 1033 / 500',
     'have h := Real.sqrt_pos.mpr (show (0 : ℝ) < 15 by norm_num)\nhave h2 := Real.sq_sqrt (show (0 : ℝ) ≤ 15 by norm_num)\n'
     'rw [div_lt_iff₀ h]\nnlinarith'),
    ('1 / (2 * (1 / 4) * 1) = 2', 'norm_num'),
    ('2 * (1 / 4) * (1 / 2) / (1 - (1 / 2) ^ 2) = 1 / 3', 'norm_num'),
    ('2 * (1 / 4) * 2 / (1 - 2 ^ 2) = -1 / 3', 'norm_num'),
    ('Real.exp (-2 * (Real.log 2 / 2))',
     'rw [show -2 * (Real.log 2 / 2) = -Real.log 2 by ring, Real.exp_neg, Real.exp_log (by norm_num)]\nnorm_num'),
    ('Real.sqrt (2 * C - x ^ 2) ^ 2', 'rw [Real.sq_sqrt h]\nring'),
    ('1 / (y * (1 - y))', 'first | (field_simp; done) | (field_simp; ring)'),
    ('= x + C',
     'rw [show (1 / (1 + Real.exp (-x - C))) / (1 - 1 / (1 + Real.exp (-x - C))) = Real.exp (x + C) by\n'
     '  have hE : 0 < Real.exp (-x - C) := Real.exp_pos _\n'
     '  rw [show x + C = -(-x - C) by ring, Real.exp_neg]\n  field_simp\n  ring]\nexact Real.log_exp _'),
    ('exp (-C) * Real.exp (-x)', 'rw [show -x - C = -C + -x by ring, Real.exp_add]\nring'),
    ('sin (Real.sin (x + 2 * Real.pi))', 'rw [Real.sin_add_two_pi]'),
    ('Real.exp (Real.log x) = x', 'exact Real.exp_log hx'),
    ('0 ^ 2 / 2 - cos pi = 1', 'norm_num'),
    ('(C + exp x) * exp (-2 * x)', 'rw [add_mul, ← Real.exp_add]\nring_nf'),
    ('1 / (1 + 1 * exp (-0))', 'norm_num'),
    ('-exp (-log 2) + 2 * exp (-2 * log 2) = 0',
     'rw [show -2 * Real.log 2 = -(Real.log 2 + Real.log 2) by ring, Real.exp_neg, Real.exp_neg, Real.exp_add,\n'
     '  Real.exp_log (by norm_num)]\nnorm_num'),
    ('exp (-log 2) - exp (-2 * log 2) = 1 / 4',
     'rw [show -2 * Real.log 2 = -(Real.log 2 + Real.log 2) by ring, Real.exp_neg, Real.exp_neg, Real.exp_add,\n'
     '  Real.exp_log (by norm_num)]\nnorm_num'),
    ('exp (-2 * x) = exp (-x) ^ 2', 'rw [sq, ← Real.exp_add]\nring_nf'),
    ('2 + exp (-2 * 0) = 3', 'norm_num'),
    ('2 - exp (-2 * 0) = 1', 'norm_num'),
    ('cosh 1 ≠ 2',
     'intro h\nrw [Real.cosh_eq] at h\nhave h1 := Real.exp_one_lt_d9\n'
     'have h2 : Real.exp (-1) < 1 := by rw [← Real.exp_zero]; exact Real.exp_lt_exp.mpr (by norm_num)\nlinarith'),
    ('cos (ω₀ * t - φ) = R', 'rw [Real.cos_sub]\nring'),
    ('ω₀ * (t + 2 * pi / ω₀) - φ',
     'rw [show ω₀ * (t + 2 * Real.pi / ω₀) - φ = ω₀ * t - φ + 2 * Real.pi by field_simp; ring, Real.cos_add_two_pi]'),
    ('sqrt 19 / 2) ^ 2 = 5', 'rw [div_pow, div_pow, Real.sq_sqrt (by norm_num)]\nnorm_num'),
    ('/ (sqrt 19 / 2) = 1 / sqrt 19',
     'have h19 : Real.sqrt 19 ≠ 0 := by positivity\nfirst | (field_simp; done) | (field_simp; ring)'),
    ('ωd * (t + 2 * pi / ωd) - φ',
     'rw [show ωd * (t + 2 * Real.pi / ωd) - φ = ωd * t - φ + 2 * Real.pi by field_simp; ring, Real.cos_add_two_pi,\n'
     '  show -γ * (t + 2 * Real.pi / ωd) = -γ * (2 * Real.pi / ωd) + -γ * t by ring, Real.exp_add]\nring'),
    ('pi / γ) / (2 * pi / ωd)', 'first | (field_simp; done) | (field_simp; ring)'),
    ('F ^ 2 / ((ω₀ ^ 2 - ω ^ 2) ^ 2', 'first | (field_simp; done) | (field_simp; ring)'),
    ('= 2 * γ * ω / (ω₀ ^ 2 - ω ^ 2)', 'first | (field_simp; done) | (field_simp; ring)'),
    ('cos (ω₀ * t - pi / 2)', 'rw [Real.cos_sub_pi_div_two]'),
    ('F / (2 * γ * ω₀) =', 'first | (field_simp; done) | (field_simp; ring)'),
    ('0.935 < sqrt (7 / 8)', 'rw [Real.lt_sqrt (by norm_num)]\nnorm_num'),
    ('sqrt (7 / 8) < 0.936', "rw [Real.sqrt_lt' (by norm_num)]\nnorm_num"),
    ('1 / sqrt (15 / 64)',
     "rw [Real.sqrt_div' _ (by norm_num : (0 : ℝ) ≤ 64), show Real.sqrt 64 = 8 by\n"
     "  rw [show (64 : ℝ) = 8 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]]\n"
     'have h15 : Real.sqrt 15 ≠ 0 := by positivity\nfirst | (field_simp; done) | (field_simp; ring)'),
    ('2.065 < 8 / sqrt 15',
     'rw [lt_div_iff₀ (by positivity)]\nnlinarith [Real.sq_sqrt (show (0 : ℝ) ≤ 15 by norm_num), Real.sqrt_nonneg 15]'),
    ('8 / sqrt 15 < 2.066',
     'rw [div_lt_iff₀ (by positivity)]\nnlinarith [Real.sq_sqrt (show (0 : ℝ) ≤ 15 by norm_num), Real.sqrt_nonneg 15]'),
    ('tan (pi - arctan (1 / 3))', 'rw [Real.tan_pi_sub, Real.tan_arctan]\nnorm_num'),
    ('2 * γ * ω * F / ((ω₀ ^ 2 - ω ^ 2) ^ 2 + 4 * γ ^ 2 * ω ^ 2)) = F', 'first | (field_simp; done) | (field_simp; ring)'),
    ('2 * γ * ω * F / ((ω₀ ^ 2 - ω ^ 2) ^ 2 + 4 * γ ^ 2 * ω ^ 2)) = 0', 'first | (field_simp; done) | (field_simp; ring)'),
    ('sqrt (2 / pi) ^ 2 * (pi / 2) = 1', 'rw [Real.sq_sqrt (by positivity)]\nfield_simp'),
    ('cos (arccos x) = x', 'exact Real.cos_arccos h₁ h₂'),
    ('cos (2 * arccos x)', 'rw [Real.cos_two_mul, Real.cos_arccos h₁ h₂]'),
    ('F - 2 * c * I + c ^ 2 * P', 'first | (field_simp; done) | (field_simp; ring)'),
    ("-x * (-(x * u'' + u')", 'first | (field_simp; done) | (field_simp; ring)'),
    ('32 / pi < 0.002',
     'have hp := Real.pi_pos\nhave h1 := Real.pi_lt_d6\n'
     'have h6 : Real.pi ^ 6 < 3.141593 ^ 6 := pow_lt_pow_left₀ h1 hp.le (by norm_num)\nnorm_num at h6\n'
     'rw [show Real.pi ^ 5 / 30 - 32 / Real.pi = (Real.pi ^ 6 - 960) / (30 * Real.pi) by field_simp; ring,\n'
     '  div_lt_iff₀ (by positivity)]\nnlinarith'),
    ("r ^ 2 - 2 * r * r' * c + r' ^ 2", 'first | (field_simp; done) | (field_simp; ring)'),
    ('(480965 / 200000 / 2) ^ 2', 'norm_num'),
    ('(1202413 / 500000 / 2) ^ 2', 'norm_num'),
    ('pi ^ 5 / 30 - 32 / pi < 1 / 500 * (pi ^ 5 / 30)',
     'have h2 := Real.pi_lt_d6\nhave hp := Real.pi_pos\n'
     'have h6 : Real.pi ^ 6 < (3.141593 : ℝ) ^ 6 := pow_lt_pow_left₀ h2 hp.le (by norm_num)\n'
     'have e : Real.pi ^ 5 / 30 - 32 / Real.pi - 1 / 500 * (Real.pi ^ 5 / 30) = '
     '(499 * Real.pi ^ 6 - 480000) / (15000 * Real.pi) := by\n  field_simp\n  ring\n'
     'have hn : (499 * Real.pi ^ 6 - 480000) / (15000 * Real.pi) < 0 := '
     'div_neg_of_neg_of_pos (by norm_num at h6; linarith) (by positivity)\nlinarith'),
    ('(2.404826 / 2) ^ 2', 'norm_num'),
    ('exp (-(x - c * 0) ^ 2)', 'simp'),
    ('(sin (x + c * t) - sin (x - c * t)) / (2 * c)', 'rw [Real.sin_add, Real.sin_sub]\nfield_simp\nring'),
    ("2 * f (r - t) / r ^ 3", 'first | (field_simp; done) | (field_simp; ring)'),
    ("= -2 * f' (r - t) / r", 'first | (field_simp; done) | (field_simp; ring)'),
    ('r ^ 2 * (F / r) ^ 2', 'field_simp'),
    ('sqrt (k ^ 2 + m ^ 2) / k * (k /',
     'have hs : Real.sqrt (k ^ 2 + m ^ 2) ≠ 0 := (Real.sqrt_pos.mpr h).ne\'\nfield_simp'),
    ('exp (-0 ^ 2 / (4 * κ * t))', 'simp'),
    ('sqrt (2 * κ * (4 * t))',
     'rw [show 2 * κ * (4 * t) = 2 ^ 2 * (2 * κ * t) by ring, Real.sqrt_mul (by norm_num), Real.sqrt_sq (by norm_num)]'),
    ("T' / (κ * T) = X'' / X", 'field_simp\nlinear_combination h'),
    ('sin pi + exp (-9 * t) * sin (3 * pi)',
     'have h3 : Real.sin (3 * Real.pi) = 0 := by exact_mod_cast Real.sin_nat_mul_pi 3\n'
     'have h5 : Real.sin (5 * Real.pi) = 0 := by exact_mod_cast Real.sin_nat_mul_pi 5\nsimp [h3, h5]'),
    ('sin (n * x) * cos (n * t) =',
     'rw [show n * (x - t) = n * x - n * t by ring, show n * (x + t) = n * x + n * t by ring, Real.sin_sub,\n'
     '  Real.sin_add]\nring'),
    ('cos x * sin (c * 0) / c', 'simp'),
    ('sqrt pi / (sqrt (pi * κ * t)',
     'rw [← Real.sqrt_mul (by positivity : (0 : ℝ) ≤ Real.pi * κ * t) (1 / (κ * t)),\n'
     '  show Real.pi * κ * t * (1 / (κ * t)) = Real.pi by field_simp, div_self (Real.sqrt_pos.mpr Real.pi_pos).ne\']'),
    ('m ^ 2 - sqrt (K + m ^ 2) ^ 2 + K = 0', 'rw [Real.sq_sqrt h]\nring'),
    ('sin (2 * (x - c * t)) + sin (2 * (x + c * t))) / 2',
     'rw [show 2 * (x - c * t) = 2 * x - 2 * (c * t) by ring, show 2 * (x + c * t) = 2 * x + 2 * (c * t) by ring,\n'
     '  Real.sin_sub, Real.sin_add]\nring'),
    ('(cos (2 * (x - c * t)) - cos (2 * (x + c * t))) / (4 * c) =',
     'rw [show 2 * (x - c * t) = 2 * x - 2 * (c * t) by ring, show 2 * (x + c * t) = 2 * x + 2 * (c * t) by ring,\n'
     '  Real.cos_sub, Real.cos_add]\nfield_simp\nring'),
    ('sqrt pi * sqrt (4 * κ * t) = sqrt (4 * pi * κ * t)', 'rw [← Real.sqrt_mul Real.pi_pos.le]\nring_nf'),
    ('4 * pi * r ^ 2 * (1 / (4 * pi * r ^ 2)) = 1', 'field_simp'),
    ('1 / (2 * pi * r) * (2 * pi * r) = 1', 'field_simp'),
    ('-r ^ (2 - 3) / ((3 - 2) * (4 * pi))',
     'rw [show (2 : ℝ) - 3 = -1 by norm_num, Real.rpow_neg_one]\nfield_simp\nring'),
    ('(1 - x) * (sin (pi * x) / pi ^ 2', 'first | (field_simp; done) | (field_simp; ring)'),
    ('sin x * (-cos (1 - x)) - cos x * sin (1 - x) = -sin 1',
     'rw [Real.cos_sub, Real.sin_sub]\nlinear_combination (-Real.sin 1) * Real.sin_sq_add_cos_sq x'),
    ('/ sin 1 = -1',
     'rw [show Real.sin ξ * -Real.cos (1 - ξ) - Real.cos ξ * Real.sin (1 - ξ) = -Real.sin 1 by\n'
     '  rw [Real.cos_sub, Real.sin_sub]; linear_combination (-Real.sin 1) * Real.sin_sq_add_cos_sq ξ]\nfield_simp'),
    ('(ua * ub\' - ua\' * ub) / (-(p *', 'first | (field_simp; done) | (field_simp; ring)'),
    ('exp (-1) * (1 + 1) - 92 / 125',
     'have h1 := Real.exp_one_gt_d9\nhave h2 := Real.exp_one_lt_d9\nhave he := Real.exp_pos 1\n'
     'have hi : Real.exp 1 * (Real.exp 1)⁻¹ = 1 := mul_inv_cancel₀ he.ne\'\n'
     'rw [Real.exp_neg, abs_lt]\nconstructor <;> nlinarith [inv_pos.mpr he]'),
    ('exp (-5) * (1 + 5) - 1 / 25',
     'have h1 := Real.exp_one_gt_d9\nhave h2 := Real.exp_one_lt_d9\n'
     'have e5 : Real.exp 5 = Real.exp 1 ^ 5 := by rw [← Real.exp_nat_mul]; norm_num\n'
     'have hl : (2.7182818283 : ℝ) ^ 5 < Real.exp 1 ^ 5 := pow_lt_pow_left₀ h1 (by norm_num) (by norm_num)\n'
     'have hu : Real.exp 1 ^ 5 < (2.7182818286 : ℝ) ^ 5 := pow_lt_pow_left₀ h2 (by positivity) (by norm_num)\n'
     'norm_num at hl hu\nhave he := Real.exp_pos 5\n'
     'have hi : Real.exp 5 * (Real.exp 5)⁻¹ = 1 := mul_inv_cancel₀ he.ne\'\n'
     'rw [Real.exp_neg, abs_lt, e5] at *\nconstructor <;> nlinarith [inv_pos.mpr he]'),
    ('exp (-(1 / 10)) * (1 + 1 / 10) - 199 / 200',
     'have hb := Real.exp_bound (x := -(1 / 10)) (by norm_num) (n := 4) (by norm_num)\n'
     'norm_num [Finset.sum_range_succ, Nat.factorial] at hb\nrw [abs_le] at hb\nrw [abs_lt]\n'
     'constructor <;> nlinarith [hb.1, hb.2]'),
    ('ω - i * ε / (2 * ω)) ^ 2',
     'have h : (i * ε / (2 * ω)) ^ 2 = -(ε ^ 2 / (4 * ω ^ 2)) := by rw [div_pow, mul_pow, hi]; ring\n'
     'rw [sub_sq, h]\nfield_simp\nring'),
    ('-i * (-E / (ω + ω))', 'first | (field_simp; done) | (field_simp; ring)'),
    ('-i * (-(c - i * s) / (2 * ω)', 'linear_combination (-s / ω) * hi'),
    ('s / ω - (s / (2 * ω) + i * c / (2 * ω))', 'linear_combination (s / (2 * ω)) * hi'),
    ('0 - (-s / (2 * ω) + i * c / (2 * ω))', 'linear_combination (s / (2 * ω)) * hi'),
    ('i * (E / (i * ω + i * ω))', 'have hi0 : i ≠ 0 := by rintro rfl; norm_num at hi\nfield_simp\nring'),
    ("1 / r ^ 2 * (r * v'' + v' - v')", 'first | (field_simp; done) | (field_simp; ring)'),
    ('sqrt (1 - 9 / 400) - 989 / 1000',
     "rw [abs_lt]\nconstructor\n· rw [lt_sub_iff_add_lt', Real.lt_sqrt (by norm_num)]; norm_num\n"
     "· rw [sub_lt_iff_lt_add', Real.sqrt_lt' (by norm_num)]; norm_num"),
    ('ξ / pi * ((pi - ξ) * cos (n * ξ) / n', 'first | (field_simp; done) | (field_simp; ring)'),
    ('x ^ 2 * (pi - x) / (2 * pi) + x * (pi - x) ^ 2 / (2 * pi)', 'first | (field_simp; done) | (field_simp; ring)'),
    ('1 / 2 * (8 / (pi * n ^ 3))', 'first | (field_simp; done) | (field_simp; ring)'),
    ('pi / 4 * b ^ 2 * (sin t ^ 2 + cos t ^ 2) = pi / 4 * b ^ 2', 'rw [Real.sin_sq_add_cos_sq, mul_one]'),
    ('1 / (2 * pi ^ 2 * r) * (pi / 2 * exp (-m * r))', 'first | (field_simp; done) | (field_simp; ring)'),
    ('cos (ω * t) - cos (ω₀ * t) = 2 * sin',
     'rw [Real.cos_sub_cos, show (ω * t - ω₀ * t) / 2 = -((ω₀ - ω) * t / 2) by ring, Real.sin_neg,\n'
     '  show (ω * t + ω₀ * t) / 2 = (ω₀ + ω) * t / 2 by ring]\nring'),
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
