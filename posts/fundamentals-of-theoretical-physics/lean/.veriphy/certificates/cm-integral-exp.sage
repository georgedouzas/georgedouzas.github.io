# name: cm-integral-exp
# technique: fundamental theorem of calculus with an antiderivative from Sage (definite_integral recipe), for
#   integrands built from exp, polynomials, sin and cos of a linear argument, with symbolic parameters
# goal: ∫ x in a..b, f x = v, f elementary (exp (c x), polynomials, sin, cos), with the nonzero conditions of the
#   parameters stated as hypotheses
# description: Sage integrates f, checks the antiderivative F by differentiation and F b - F a = v; Lean proves
#   HasDerivAt F f on the interval from mathlib's composed derivative lemmas, applies
#   intervalIntegral.integral_eq_sub_of_hasDerivAt and evaluates the bounds with field_simp and ring.
INDEX = 0
I = problem.integrals[INDEX]
assert I.body is not None and I.lower is not None and I.upper is not None, 'no readable interval integral'
X = SR.var('x_')
body = SR(I.body).subs({I.var: X})
F = integrate(body, X)
assert not str(F).count('integrate'), f'no elementary antiderivative of {body}'
assert bool((diff(F, X) - body).simplify_full() == 0), 'the antiderivative does not differentiate back'
value = (F.subs({X: I.upper}) - F.subs({X: I.lower}))
assert goal.expr is not None and bool(SR(goal.expr).simplify_full() == 0), f'the integral is {value.simplify_full()}, not the goal value'
v = str(symbolic(I.var, problem, lean_type='')).strip('()')
F_lean = str(symbolic(F.subs({X: I.var}), problem, lean_type=''))
term = has_deriv_at(F.subs({X: I.var}), I.var, I.var, problem)
lines = [
    f'have hF : ∀ {v} ∈ Set.uIcc ({I.lower_text} : ℝ) ({I.upper_text} : ℝ), HasDerivAt (fun {v} => ({F_lean} : ℝ)) (({I.function}) {v}) {v} := by',
    f'  intro {v} _',
    f'  refine (({term}).congr_of_eventuallyEq (Filter.Eventually.of_forall fun y => ?_)).congr_deriv ?_',
    '  · beta_reduce; first | rfl | ring1 | (simp; ring1) | (simp; done) | (field_simp; ring1) | (ring_nf; done)',
    '  · beta_reduce; first | ring1 | (field_simp; done) | (field_simp; ring1) | (field_simp; ring_nf; done) | (simp; ring1) | (ring_nf; done)',
    'rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hF (by apply Continuous.intervalIntegrable; fun_prop)]',
    'all_goals (try beta_reduce)',
    'all_goals (try simp only [mul_zero, zero_mul, zero_div, neg_zero, Real.exp_zero, sub_zero, zero_sub, add_zero, zero_add, Real.sin_zero, Real.cos_zero, mul_one, one_mul])',
    'all_goals (first | ring1 | (field_simp; ring1) | (field_simp; ring_nf; done) | (norm_num; done) | (ring_nf; done))',
]
proof = tactic('\n'.join(lines))
