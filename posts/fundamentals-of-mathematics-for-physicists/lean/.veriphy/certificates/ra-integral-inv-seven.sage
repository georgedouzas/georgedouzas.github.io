# name: ra-integral-inv-seven
# technique: fundamental theorem of calculus with an antiderivative from Sage
# goal: ∫₁⁷ dx/x = log 7
# description: Sage finds F with F′ = f and checks F b - F a = v; Lean proves HasDerivAt F f and evaluates the bounds.
INDEX = 0  # EDIT: which interval integral of the goal, when it has several
# the recipe's emitter, with the side condition x ≠ 0 discharged from x ∈ [1, 7] (named here `hx_veriphy`)
from sage.all import SR, integrate
from veriphy.toolkit._integral import VARIABLE, DERIVATIVE_CLOSE, FUNCTION_CLOSE, VALUE_CLOSE, _lean, _endpoint_facts
integral = problem.integrals[INDEX]
x = SR.var(VARIABLE)
body = integral.body.subs({integral.var: x})
antiderivative = integrate(body, x)
assert bool((antiderivative.diff(x) - body).simplify_full() == 0), 'the antiderivative does not differentiate back'
assert bool(SR(goal.expr).simplify_full() == 0), 'the integral is not the stated value'
variable = VARIABLE[:-1]
NONZERO = {f'{variable} ≠ 0': f'exact hne {variable} hx_veriphy'}
f_text = f'fun {variable} => {_lean(antiderivative, problem)}'
derivative = has_deriv_at(antiderivative, x, x, problem, proofs=NONZERO)
interval = f'Set.uIcc ({integral.lower_text} : ℝ) ({integral.upper_text} : ℝ)'
bounds = [(integral.lower, integral.lower_text), (integral.upper, integral.upper_text)]
statement = f'HasDerivAt ({f_text}) (({integral.function}) {variable}) {variable}'
lines = ['have hne : ∀ x ∈ Set.uIcc (1 : ℝ) 7, x ≠ 0 := by', '  intro x hx', '  rw [Set.uIcc_of_le (by norm_num)] at hx',
         "  exact (lt_of_lt_of_le zero_lt_one hx.1).ne'",
         f'have hF : ∀ {variable} ∈ {interval}, {statement} := by', f'  intro {variable} hx_veriphy', '  beta_reduce',
         f'  refine (({derivative}).congr_of_eventuallyEq (Filter.Eventually.of_forall fun y => ?_)).congr_deriv ?_',
         f'  · {FUNCTION_CLOSE}', f'  · {DERIVATIVE_CLOSE}',
         f"rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := {f_text}) (f' := {integral.function}) hF",
         '  (ContinuousOn.intervalIntegrable (ContinuousOn.div continuousOn_const continuousOn_id hne))]', 'beta_reduce']
facts = _endpoint_facts(antiderivative, x, bounds, problem)
lines += facts
if facts:
    lines.append(f'simp only [{", ".join(f"he{i}" for i in range(len(facts)))}]')
lines.append(VALUE_CLOSE)
proof = tactic('\n'.join(lines))
