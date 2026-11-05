# name: ra-fubini-inner
# technique: fundamental theorem of calculus with an antiderivative from Sage
# goal: ∫₀¹ (x² - y²)/(x² + y²)² dy = 1/(1 + x²) for x > 0
# description: Sage finds the antiderivative y/(x² + y²) and checks F(1) - F(0); Lean proves HasDerivAt and evaluates the bounds.
INDEX = 0  # the single interval integral of the goal
# the recipe's emitter, with the side condition x² + y² ≠ 0 (x > 0 by hypothesis) discharged for every y of the interval
from sage.all import SR, integrate
from veriphy.toolkit._integral import VARIABLE, DERIVATIVE_CLOSE, FUNCTION_CLOSE, VALUE_CLOSE, _lean, _endpoint_facts
integral = problem.integrals[INDEX]
x = SR.var(VARIABLE)
body = integral.body.subs({integral.var: x})
antiderivative = integrate(body, x)
assert bool((antiderivative.diff(x) - body).simplify_full() == 0), 'the antiderivative does not differentiate back'
assert bool(SR(goal.expr).simplify_full() == 0), 'the integral is not the stated value'
variable = VARIABLE[:-1]
NONZERO = {f'x ^ 2 + ({variable} ^ 2) ≠ 0': f'exact hne {variable} hx_veriphy'}
f_text = f'fun {variable} => {_lean(antiderivative, problem)}'
derivative = has_deriv_at(antiderivative, x, x, problem, proofs=NONZERO)
interval = f'Set.uIcc ({integral.lower_text} : ℝ) ({integral.upper_text} : ℝ)'
bounds = [(integral.lower, integral.lower_text), (integral.upper, integral.upper_text)]
statement = f'HasDerivAt ({f_text}) (({integral.function}) {variable}) {variable}'
lines = [f'have hne : ∀ y ∈ {interval}, x ^ 2 + y ^ 2 ≠ 0 := fun y _ => by positivity',
         f'have hF : ∀ {variable} ∈ {interval}, {statement} := by', f'  intro {variable} hx_veriphy',
         f'  have hx0 : x ^ 2 + {variable} ^ 2 ≠ 0 := hne {variable} hx_veriphy', '  beta_reduce',
         f'  refine (({derivative}).congr_of_eventuallyEq (Filter.Eventually.of_forall fun y => ?_)).congr_deriv ?_',
         f'  all_goals first | ({FUNCTION_CLOSE}) | ({DERIVATIVE_CLOSE}) | (field_simp; ring_nf; done)',
         f"rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := {f_text}) (f' := {integral.function}) hF",
         '  (ContinuousOn.intervalIntegrable (ContinuousOn.div (by fun_prop) (by fun_prop) (fun y hy => pow_ne_zero 2 (hne y hy))))]',
         'try beta_reduce']
facts = _endpoint_facts(antiderivative, x, bounds, problem)
lines += facts
if facts:
    lines.append(f'simp only [{", ".join(f"he{i}" for i in range(len(facts)))}]')
lines.append(VALUE_CLOSE + ' | (norm_num; ring1) | (field_simp; ring_nf; done)')
proof = tactic('\n'.join(lines))
