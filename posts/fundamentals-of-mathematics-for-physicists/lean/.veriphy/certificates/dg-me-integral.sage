# name: dg-me-integral
# technique: fundamental theorem of calculus with an antiderivative from Sage (definite_integral recipe), with the
#   side conditions of square roots and real powers discharged by positivity and the closing steps widened
# goal: ∫ x in a..b, f x = v, f elementary (polynomials, sin, cos, sqrt, powers), possibly with symbolic constants
# description: Sage finds F with F' = f and checks F b - F a = v; Lean proves HasDerivAt F f (the nonzero conditions
#   `u ^ 2 + 1 ≠ 0` of a square root are proved by positivity) and evaluates the bounds, closing by ring1, field_simp,
#   norm_num or the simplification of `(2 : ℝ) ^ (3 / 2)` into `2 √2`.
from veriphy.toolkit._integral import _lean, _endpoint_facts, VARIABLE, FUNCTION_CLOSE, DERIVATIVE_CLOSE, VALUE_CLOSE
INDEX = 0
PROOFS = {}
I = problem.integrals[INDEX]
for _v in set(SR(I.lower).variables()) | set(SR(I.upper).variables()):
    assume(_v > 0); assume(_v < pi)
x = SR.var(VARIABLE)
body = I.body.subs({I.var: x})
F = integrate(body, x)
assert 'integrate' not in str(F), f'Sage finds no elementary antiderivative of {body}'
assert bool((F.diff(x) - body).simplify_full() == 0), 'the antiderivative does not differentiate back'
assert goal.expr is not None and bool(SR(goal.expr).simplify_full() == 0), \
    f'the integral is {(F.subs({x: I.upper}) - F.subs({x: I.lower})).simplify_full()}, not the goal value'
variable = VARIABLE[:-1]
f_text = f'fun {variable} => {_lean(F, problem)}'
proofs = dict(PROOFS)
for _ in range(6):
    try:
        derivative = has_deriv_at(F, x, x, problem, proofs=proofs)
        break
    except Exception as exc:
        import re as _re
        m = _re.search(r"needs '(.*?)'", str(exc))
        if not m or m.group(1) in proofs:
            raise
        proofs[m.group(1)] = 'positivity'
interval = f'Set.uIcc ({I.lower_text} : ℝ) ({I.upper_text} : ℝ)'
bounds = [(I.lower, I.lower_text), (I.upper, I.upper_text)]
statement = f'HasDerivAt ({f_text}) (({I.function}) {variable}) {variable}'
lines = [
    f'have hF : ∀ {variable} ∈ {interval}, {statement} := by',
    f'  intro {variable} _',
    '  beta_reduce',
    f'  refine (({derivative}).congr_of_eventuallyEq (Filter.Eventually.of_forall fun y => ?_)).congr_deriv ?_',
    f'  · {FUNCTION_CLOSE}',
    f'  · {DERIVATIVE_CLOSE} | (field_simp; done) | (simp; field_simp; ring1)',
    f"rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := {f_text}) (f' := {I.function}) hF",
    '  (by apply Continuous.intervalIntegrable; fun_prop)]',
    'beta_reduce',
]
facts = _endpoint_facts(F, x, bounds, problem)
lines += facts
if facts:
    lines.append(f'simp only [{", ".join(f"he{i}" for i in range(len(facts)))}]')
lines.append('all_goals first | (norm_num; done) | ring1 | (field_simp; ring1) | (simp; done) | (norm_num; ring1) '
             '| (simp; ring1) | (field_simp; done) '
             '| (rw [show ((2 : ℝ) ^ (3 / 2 : ℝ)) = 2 * Real.sqrt 2 by\n'
             '      rw [show (3 / 2 : ℝ) = 1 + 1 / 2 by norm_num, Real.rpow_add (by norm_num), Real.rpow_one, Real.sqrt_eq_rpow]]; ring1)')
proof = tactic('\n'.join(lines))
