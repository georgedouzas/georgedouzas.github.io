# name: dg-me-trig
# technique: identities and values of sin, cos, sqrt, exp checked in Sage first, then closed in Lean by the named
#   mathlib rewrites listed for the goal (transcendental_rewrite recipe, REWRITES given per goal shape)
# goal: an identity or a value between sin, cos, sqrt, exp expressions of the metric chapter (lengths of parallels,
#   metric components at special latitudes, half-angle reductions)
# description: Sage confirms the statement exactly (simplify_full, canonicalize_radical, trig simplification); the
#   Lean proof applies the rewrites listed in HINTS for the goal's shape (special values sin (π/6), sin (π/3),
#   cos (π/2), double-angle and half-angle formulas) and closes by ring or norm_num.
HINTS = [
    # (a substring of the goal's Lean text, the tactic closing it)
    ('sin (pi / 6) ^ 2 = 1 / 4', 'rw [Real.sin_pi_div_six]\nnorm_num'),
    ('sin (pi / 2) ^ 2 = 1', 'rw [Real.sin_pi_div_two]\nnorm_num'),
    ('sin (pi / 2) = 1', 'exact Real.sin_pi_div_two'),
    ('cos (pi / 2) = 0', 'exact Real.cos_pi_div_two'),
    ('1 / sin (pi / 6) ^ 2 = 4', 'rw [Real.sin_pi_div_six]\nnorm_num'),
    ('2 * pi * sin (pi / 3) = sqrt 3 * pi', 'rw [Real.sin_pi_div_three]\nring'),
    ('2 * pi * sin (pi / 6) = pi', 'rw [Real.sin_pi_div_six]\nring'),
    ('2 * pi * sin (pi / 2) = 2 * pi', 'rw [Real.sin_pi_div_two]\nring'),
    ('cos (pi / 6) ^ 2 / (1 + sin (pi / 6) ^ 2) = 3 / 5', 'rw [Real.cos_pi_div_six, Real.sin_pi_div_six]\nnorm_num'),
    ('cos (pi / 4) ^ 2 / (1 + sin (pi / 4) ^ 2) = 1 / 3',
     'rw [Real.cos_pi_div_four, Real.sin_pi_div_four]\nhave h := Real.sq_sqrt (show (0 : ℝ) ≤ 2 by norm_num)\nrw [div_pow, div_pow, h]\nnorm_num'),
    ('cos (pi / 2) ^ 2 / (1 + sin (pi / 2) ^ 2) = 0', 'rw [Real.cos_pi_div_two]\nnorm_num'),
    ('sin (pi - θ) = sin θ', 'exact Real.sin_pi_sub θ'),
    ('sin (pi - x) ^ 2 = sin x ^ 2', 'rw [Real.sin_pi_sub]'),
    ('cos (θ / 2) ^ 2 + sin (θ / 2) ^ 2 = 1', 'exact Real.cos_sq_add_sin_sq _'),
    ('1 / 2 * cos (θ / 2) ^ 2 - 1 / 4 = 1 / 4 * cos θ',
     'rw [show θ = 2 * (θ / 2) by ring, Real.cos_two_mul]\nring_nf'),
    ('cos θ ^ 2 - cos (2 * θ) =', 'rw [Real.cos_two_mul]\nring'),
    ('sin θ ^ 2 = 1 - cos θ ^ 2', 'rw [Real.sin_sq]'),
    ('1 - cos θ ^ 2 = sin θ ^ 2', 'rw [Real.sin_sq]'),
    ('sqrt (1 - cos θ ^ 2) = |sin θ|', 'rw [← Real.sin_sq, Real.sqrt_sq_eq_abs]'),
    ('sqrt (sin θ ^ 2) = sin θ', 'exact Real.sqrt_sq h'),
    ('sqrt (sin (pi / 6) ^ 2) = 1 / 2', 'rw [Real.sin_pi_div_six, Real.sqrt_sq (by norm_num)]'),
    ('sqrt |R ^ 2| = R', 'rw [abs_of_nonneg (sq_nonneg R)]\nexact Real.sqrt_sq h'),
    ('exp (2 * u) = exp u ^ 2', 'rw [sq, ← Real.exp_add]\nring_nf'),
    ('cos t ^ 2 + (-sin t) ^ 2 = 1', 'rw [neg_sq]\nexact Real.cos_sq_add_sin_sq t'),
    ('(-(r * sin t)) ^ 2 + (r * cos t) ^ 2 = r ^ 2', 'linear_combination r ^ 2 * Real.sin_sq_add_cos_sq t'),
    ('((1 + sqrt 5) / 2) ^ 2 - (1 + sqrt 5) / 2 - 1 = 0', 'nlinarith [Real.sq_sqrt (show (0 : ℝ) ≤ 5 by norm_num)]'),
    ('((1 - sqrt 5) / 2) ^ 2 - (1 - sqrt 5) / 2 - 1 = 0', 'nlinarith [Real.sq_sqrt (show (0 : ℝ) ≤ 5 by norm_num)]'),
    ('sqrt (sin θ ^ 2 / 64) = sin θ / 8',
     'rw [show Real.sin θ ^ 2 / 64 = (Real.sin θ / 8) ^ 2 by ring]\nexact Real.sqrt_sq (by positivity)'),
    ('sqrt (r ^ 2) = r', 'exact Real.sqrt_sq h'),
    ('sqrt (R ^ 2) = R', 'exact Real.sqrt_sq h'),
    ('sqrt (R ^ 4 * sin θ ^ 2) = R ^ 2 * sin θ',
     'rw [show R ^ 4 * Real.sin θ ^ 2 = (R ^ 2 * Real.sin θ) ^ 2 by ring]\nexact Real.sqrt_sq (mul_nonneg (sq_nonneg R) h)'),
    ('sqrt (16 / (1 + u ^ 2 + v ^ 2) ^ 4) = 4 / (1 + u ^ 2 + v ^ 2) ^ 2',
     'rw [show (16 : ℝ) / (1 + u ^ 2 + v ^ 2) ^ 4 = (4 / (1 + u ^ 2 + v ^ 2) ^ 2) ^ 2 by ring]\nexact Real.sqrt_sq (by positivity)'),
    ('sqrt (exp (4 * u)) = exp (2 * u)',
     'rw [show Real.exp (4 * u) = Real.exp (2 * u) ^ 2 by rw [← Real.exp_nat_mul]; ring_nf]\nexact Real.sqrt_sq (Real.exp_pos _).le'),
    ('exp (8 * A) = (exp (2 * A)) ^ 4', 'rw [← Real.exp_nat_mul]\nring_nf'),
    ('sqrt (1 - v ^ 2) * T = T * sqrt (1 - v ^ 2)', 'ring'),
    ('cosh ξ ^ 2 - sinh ξ ^ 2 = 1', 'exact Real.cosh_sq_sub_sinh_sq ξ'),
    ('sinh ψ ^ 2 - cosh ψ ^ 2 = -1', 'linear_combination -(Real.cosh_sq_sub_sinh_sq ψ)'),
    ('sqrt 2 * pi < 2 * pi',
     'have h2 : Real.sqrt 2 < 2 := by\n  rw [Real.sqrt_lt\' (by norm_num)]\n  norm_num\nnlinarith [Real.pi_pos]'),
    ('sqrt (1 / 2 ^ 2 + (1 / 2 * cos t) ^ 2)', 'ring_nf'),
]
e = goal.expr
assert e is not None, 'the goal has no symbolic form; check the problem with print(goal)'
for h in problem.hyps:
    if h.expr is not None and h.rel in ('lt', 'le'):
        assume(h.expr < 0 if h.rel == 'lt' else h.expr <= 0)


def _forms(x):
    out = []
    for f in (lambda y: y.simplify_full(), lambda y: y.canonicalize_radical(),
              lambda y: y.simplify_trig().simplify_full(), lambda y: y.expand_trig().simplify_full(),
              lambda y: y.trig_expand(full=True).simplify_full(), lambda y: y.trig_reduce().simplify_full(),
              lambda y: y.expand().trig_reduce().simplify_full(),
              lambda y: y.simplify_full().canonicalize_radical(),
              lambda y: y.rewrite('cosh(x)->exp(x)').rewrite('sinh(x)->exp(x)').simplify_full() if hasattr(y, 'rewrite') else y):
        try:
            out.append(f(SR(x)))
        except Exception:
            pass
    return out


forms = _forms(e)
if goal.rel == 'ne':
    assert all(bool(f != 0) for f in forms[:1]), f'Sage cannot confirm the inequation: lhs - rhs = {forms[0]}'
elif goal.rel in ('lt', 'le'):
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
