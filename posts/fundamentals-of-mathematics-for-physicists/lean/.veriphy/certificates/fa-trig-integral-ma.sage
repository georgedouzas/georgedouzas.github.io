# name: fa-trig-integral-ma
# technique: fundamental theorem of calculus with an antiderivative from Sage (definite_integral recipe), adapted:
#   the antiderivative keeps its multiple angles sin (k x), cos (k x) (few terms for a squared partial sum), and the
#   derivative step expands every multiple angle of F' and f into sin x, cos x and closes modulo sin² + cos² = 1
# goal: ∫ x in a..b, f x = v, f a polynomial in x, sin (k x), cos (k x) (integer k) with rational and 1/π coefficients
#   (the L² error of a partial Fourier sum); bounds multiples of π
# description: Sage integrates f (the antiderivative F in multiple angles); Lean proves HasDerivAt F f with mathlib's
#   derivative lemmas, rewrites each sin (k x), cos (k x) of F' and f by the addition formulas, and closes F' = f by
#   linear_combination K * (sin² x + cos² x = 1) with the cofactor K Sage computes.
import re
from sage.all import SR, integrate, pi, PolynomialRing, QQ, sin, cos, FractionField
from veriphy.toolkit._integral import _endpoint_facts, _lean, VARIABLE, FUNCTION_CLOSE
INDEX = 0
integral = problem.integrals[INDEX]
assert integral.body is not None and integral.lower is not None and integral.upper is not None, 'no symbolic integrand'
x = SR.var(VARIABLE)
v = VARIABLE[:-1]
body = integral.body.subs({integral.var: x})
F = integrate(body, x).expand()
assert 'integrate' not in str(F), f'no elementary antiderivative of {body}'
value = (F.subs({x: integral.upper}) - F.subs({x: integral.lower})).simplify_full()
assert bool((goal.expr).simplify_full() == 0), f'the integral is {value}, not the goal value'
# multiple angles of the integrand
def kernels(e):
    out = [e]
    for o in e.operands():
        out += kernels(o)
    return out
ks = set()
for k in kernels(body) + kernels(F):
    op = k.operator()
    name = getattr(op, 'name', lambda: '')() if op is not None else ''
    if name in ('sin', 'cos'):
        m = (k.operands()[0] / x).simplify_full()
        assert m.is_integer() and m >= 1, f'argument {k.operands()[0]} is not a positive integer multiple of x'
        ks.add(int(m))
kmax = max(ks) if ks else 1
facts = []
for k in range(2, kmax + 1):
    ck = _lean(cos(k * x).expand_trig().expand(), problem)
    sk = _lean(sin(k * x).expand_trig().expand(), problem)
    if k == 2:
        split = f'show (2 : ℝ) * {v} = {v} + {v} by ring'
        facts.append(f'have hmc2 : Real.cos (2 * {v}) = {ck} := by\n    rw [{split}, Real.cos_add]; ring')
        facts.append(f'have hms2 : Real.sin (2 * {v}) = {sk} := by\n    rw [{split}, Real.sin_add]; ring')
    else:
        split = f'show ({k} : ℝ) * {v} = {k - 1} * {v} + {v} by ring'
        facts.append(f'have hmc{k} : Real.cos ({k} * {v}) = {ck} := by\n    rw [{split}, Real.cos_add, hmc{k - 1}, hms{k - 1}]; ring')
        facts.append(f'have hms{k} : Real.sin ({k} * {v}) = {sk} := by\n    rw [{split}, Real.sin_add, hmc{k - 1}, hms{k - 1}]; ring')
names = [f'hm{t}{k}' for k in range(2, kmax + 1) for t in ('c', 's')]
# the cofactor K of F' - f modulo s^2 + c^2 - 1
s_, c_, p_ = SR.var('s_veriphy_'), SR.var('c_veriphy_'), SR.var('p_veriphy_')
others = sorted({str(y) for y in (F.variables() + body.variables())} - {str(x)})
# π (and 1/π) may appear in the coefficients: clear the powers of π and work over ℚ[s, c, x, π]
R = PolynomialRing(QQ, ['s_veriphy_', 'c_veriphy_', 'x_veriphy_', 'p_veriphy_'] + others)
NPI = 8
def poly(e):
    e = e.expand_trig().expand().subs({sin(x): s_, cos(x): c_}).subs({pi: p_})
    return R((e * p_ ** NPI).expand())
D = poly(F.diff(x)) - poly(body)
rel = R('s_veriphy_^2 + c_veriphy_^2 - 1')
K, r = D.quo_rem(rel)
assert r == 0, f'the derivative differs from the integrand by {r} modulo sin² + cos² = 1'
Ksr = (SR(str(K)) / p_ ** NPI).expand().subs({s_: sin(x), c_: cos(x), p_: pi})
# with π in the coefficients the derivative step leaves π * π⁻¹, which ring cannot cancel: clear it with field_simp
has_pi = bool(F.has(pi)) or bool(body.has(pi))
norm = '(norm := first | ring1 | (field_simp; try ring1)) ' if has_pi else ''
pre_close = 'have hπ : Real.pi ≠ 0 := Real.pi_ne_zero\n    ' if has_pi else ''
if K == 0:
    close = pre_close + ('first | ring1 | (field_simp; try ring1)' if has_pi else 'ring1')
else:
    close = pre_close + f'linear_combination {norm}({_lean(Ksr, problem)}) * Real.sin_sq_add_cos_sq {v}'
rewrite = f'simp only [{", ".join(names)}]\n    ' if names else ''
f_text = f'fun {v} => {_lean(F, problem)}'
from veriphy.toolkit._derivative import has_deriv_at
derivative = has_deriv_at(F, x, x, problem)
interval = f'Set.uIcc ({integral.lower_text} : ℝ) ({integral.upper_text} : ℝ)'
statement = f'HasDerivAt ({f_text}) (({integral.function}) {v}) {v}'
lines = [f'have hF : ∀ {v} ∈ {interval}, {statement} := by', f'  intro {v} _', '  beta_reduce']
lines += ['  ' + f.replace('\n', '\n  ') for f in facts]
lines += [
    f'  refine (({derivative}).congr_of_eventuallyEq (Filter.Eventually.of_forall fun y => ?_)).congr_deriv ?_',
    f'  · {FUNCTION_CLOSE}',
    f'  · (try simp only [Pi.pow_apply, Pi.mul_apply])\n    {rewrite}{close}',
    f"rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := {f_text}) (f' := {integral.function}) hF",
    '  (by apply Continuous.intervalIntegrable; fun_prop)]',
    'beta_reduce',
]
bounds = [(integral.lower, integral.lower_text), (integral.upper, integral.upper_text)]
efacts = _endpoint_facts(F, x, bounds, problem)
# negative multiples of π at the bounds (the stock facts cover k ≥ 0)
for kern in kernels(F):
    op = kern.operator()
    name = getattr(op, 'name', lambda: '')() if op is not None else ''
    if name not in ('sin', 'cos'):
        continue
    arg = kern.operands()[0]
    for bound, text in bounds:
        k = (arg.subs({x: bound}) / pi).simplify_full()
        if not (k.is_integer() and k < 0):
            continue
        term = re.sub(rf'\b{v}\b', f'({text})', _lean(arg, problem))
        rw = f'rw [show ({term} : ℝ) = -((({-k} : ℕ) : ℝ) * Real.pi) by push_cast; ring'
        if name == 'sin':
            fact = f'have he{len(efacts)} : Real.sin ({term}) = 0 := by\n  {rw}, Real.sin_neg, Real.sin_nat_mul_pi]; simp'
        else:
            fact = f'have he{len(efacts)} : Real.cos ({term}) = {(-1) ** int(-k)} := by\n  {rw}, Real.cos_neg, Real.cos_nat_mul_pi]; norm_num'
        if all(fact.split(':', 1)[1] != f.split(':', 1)[1] for f in efacts):
            efacts.append(fact)
# `rw [...]; tac` on one line is re-flowed by Lean's formatter as `rw [...];⏎ tac`, which does not re-parse:
# put the closing tactic on its own line
efacts = [re.sub(r'\]; (exact|simp|norm_num)', r']\n  \1', f) for f in efacts]
lines += efacts
if efacts:
    lines.append(f'simp only [{", ".join(f"he{i}" for i in range(len(efacts)))}]')
lines.append('all_goals first | (norm_num; done) | ring1 | (field_simp; ring1) | (simp; done) | (norm_num; ring1) | (simp; ring1) | (field_simp; done)')
proof = tactic('\n'.join(lines))
