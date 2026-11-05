# name: dg-fk-rewrite
# technique: identities, values and inequations of exp, sin, cos, cosh, sinh, tan, arctan checked in Sage first, then
#   closed in Lean by the named mathlib rewrites listed for the goal (transcendental_rewrite recipe, REWRITES given
#   per goal shape as HINTS)
# goal: an identity (or a value, or ≠) between exp, sin, cos, cosh, sinh, tan, arctan expressions of the chapter
#   "Flows and Killing vectors": group laws of flows (addition formulas), values at t = 0, pullback factors
# description: Sage confirms the statement exactly (simplify_full, trig and exponential simplifications, interval
#   arithmetic for inequalities); the Lean proof applies the rewrites listed in HINTS for the goal's shape.
HINTS = [
    # (a substring of the goal's Lean text, the tactic closing it)
    ('cos (s + t)', 'simp only [Real.cos_add, Real.sin_add]\nconstructor <;> ring1'),
    ('cosh (s + t)', 'simp only [Real.cosh_add, Real.sinh_add]\nconstructor <;> ring1'),
    ('exp (-(s + t))', 'rw [neg_add, Real.exp_add]\nring1'),
    ('exp (s + t)', 'rw [Real.exp_add]\nconstructor <;> ring1'),
    ('exp (-s) * (exp (-t)', 'rw [← mul_assoc, ← Real.exp_add]\nring_nf'),
    ('exp (2 * t) * (dx ^ 2 + dy ^ 2)', 'rw [two_mul, Real.exp_add]\nring1'),
    ('x * cos 0 - y * sin 0 = x', 'simp'),
    ('exp 0 * x = x', 'simp'),
    ('cos 0 = 1', 'simp'),
    ('cosh 0', 'simp'),
    ('sin 0 = 0 ∧ sin pi = 0', 'exact ⟨Real.sin_zero, Real.sin_pi⟩'),
    ('exp t ^ 2 ≠ 1', 'intro h\nrw [sq, ← Real.exp_add, Real.exp_eq_one_iff] at h\nexact ht (by linarith)'),
    ('exp (2 * t) ≠ 1', 'intro h\nrw [Real.exp_eq_one_iff] at h\nexact ht (by linarith)'),
    ('exp (-(2 * t)) ≠ 1', 'intro h\nrw [Real.exp_eq_one_iff] at h\nexact ht (by linarith)'),
    ('sqrt (y ^ 2 + 0 ^ 2) = |y|', 'rw [zero_pow two_ne_zero, add_zero, Real.sqrt_sq_eq_abs]'),
    ('sin (pi - θ) = sin θ', 'exact Real.sin_pi_sub θ'),
    ('sin (pi - x 0) ^ 2 = sin (x 0) ^ 2', 'rw [Real.sin_pi_sub]'),
    ('tan (2 * arctan v / 2) = v', 'rw [show 2 * Real.arctan v / 2 = Real.arctan v by ring, Real.tan_arctan]'),
    ('sin φ ^ 2 + cos θ ^ 2 * cos φ ^ 2 = 1 - sin θ ^ 2 * cos φ ^ 2',
     'have h1 := Real.sin_sq_add_cos_sq θ\nhave h2 := Real.sin_sq_add_cos_sq φ\nlinear_combination h2 - cos φ ^ 2 * h1'),
    ('cos φ ^ 2 + cos θ ^ 2 * sin φ ^ 2 = 1 - sin θ ^ 2 * sin φ ^ 2',
     'have h1 := Real.sin_sq_add_cos_sq θ\nhave h2 := Real.sin_sq_add_cos_sq φ\nlinear_combination h2 - sin φ ^ 2 * h1'),
    ('sin φ ^ 2 + cos φ ^ 2 = 1', 'exact Real.sin_sq_add_cos_sq φ'),
    ('(cos θ / sin θ) ^ 2 + 1 = 1 / sin θ ^ 2', 'field_simp\nlinear_combination Real.sin_sq_add_cos_sq θ'),
    ('cosh w ^ 2 - sinh w ^ 2 = 1', 'exact Real.cosh_sq_sub_sinh_sq w'),
    ('(t * cosh w + x * sinh w) ^ 2 - (t * sinh w + x * cosh w) ^ 2 = t ^ 2 - x ^ 2',
     'have h := Real.cosh_sq_sub_sinh_sq w\nlinear_combination (t ^ 2 - x ^ 2) * h'),
    ('-sin θ * (2 * sin θ * cos θ) = -2 * cos θ * sin θ ^ 2', 'ring1'),
    ('sin (2 * arctan u) = 2 * u / (1 + u ^ 2)', 'rw [Real.sin_two_mul, Real.sin_arctan, Real.cos_arctan]\nfield_simp\nrw [Real.sq_sqrt (by positivity)]\nring1'),
    ('2 * arctan (exp (-0) * tan (θ / 2)) = θ', 'rw [neg_zero, Real.exp_zero, one_mul, Real.arctan_tan (by linarith) (by linarith)]\nring1'),
]
e = goal.expr
assert e is not None, 'the goal has no symbolic form; check the problem with print(goal)'
for h in problem.hyps:
    if h.expr is not None and h.rel in ('lt', 'le'):
        try:
            assume(h.expr < 0 if h.rel == 'lt' else h.expr <= 0)
        except Exception:
            pass
def _forms(x):
    out = []
    for f in (lambda y: y.simplify_full(), lambda y: y.canonicalize_radical(),
              lambda y: y.simplify_trig().simplify_full(), lambda y: y.expand_trig().simplify_full(),
              lambda y: y.trig_expand(full=True).simplify_full(), lambda y: y.trig_reduce().simplify_full(),
              lambda y: y.expand().trig_reduce().simplify_full(),
              lambda y: y.rewrite('cosh(x)->exp(x)').rewrite('sinh(x)->exp(x)').simplify_full() if hasattr(y, 'rewrite') else y,
              lambda y: y.exp_simplify() if hasattr(y, 'exp_simplify') else y,
              lambda y: y.simplify_full().expand().simplify_full()):
        try:
            out.append(f(SR(x)))
        except Exception:
            pass
    return out
def _check(expr, rel):
    forms = _forms(expr)
    if rel == 'ne':
        assert all(bool(f != 0) for f in forms[:1]), f'Sage cannot confirm the inequation: lhs - rhs = {forms[0]}'
    elif rel in ('lt', 'le'):
        value = RealIntervalField(200)(expr)
        assert value < 0 if rel == 'lt' else value <= 0, f'Sage refutes the inequality: lhs - rhs ∈ {value}'
    else:
        assert any(bool(f == 0) for f in forms), f'Sage cannot confirm the identity: lhs - rhs simplifies to {forms[0]}'
if goal.rel == 'and' or (hasattr(goal, 'parts') and goal.parts):
    for part in goal.parts:
        _check(part.expr, part.rel)
else:
    _check(e, goal.rel)
def _norm(t):
    for a, b in (('Real.', ''), ('rexp', 'exp'), ('√', 'sqrt '), ('π', 'pi'), (' ', '')):
        t = t.replace(a, b)
    return t
hints = [t for key, t in HINTS if _norm(key) in _norm(goal.text)]
assert hints, f'no rewrite is listed for {goal.text}; add one to HINTS'
proof = tactic(hints[0])
