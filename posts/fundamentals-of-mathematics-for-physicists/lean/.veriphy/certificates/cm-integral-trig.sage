# name: cm-integral-trig
# technique: fundamental theorem of calculus with an antiderivative from Sage, for integrands that are products of
#   sines and cosines of multiple angles (definite_integral recipe, with the multiple-angle expansions as facts)
# goal: ∫ x in a..b, f x = v, f a polynomial in x, sin (k x), cos (k x) (products such as sin mx sin nx)
# description: Sage integrates f and checks the antiderivative F by differentiation. The derivative of F is composed
#   from mathlib's lemmas; it equals f after expanding every sin (k y), cos (k y) into a polynomial in sin y, cos y
#   (facts proved from the addition formulas, k = 2, 3, …) and, where needed, a multiple of sin² y + cos² y - 1 that
#   Sage computes (linear_combination). FTC (intervalIntegral.integral_eq_sub_of_hasDerivAt) and the same expansions
#   at the bounds give the value.
INDEX = 0
import re as _re
I = problem.integrals[INDEX]
assert I.body is not None and I.lower is not None and I.upper is not None, 'no readable interval integral'
X = SR.var('x_')
body = SR(I.body).subs({I.var: X})
F = integrate(body, X)
assert not str(F).count('integrate'), f'no elementary antiderivative of {body}'
assert bool((diff(F, X) - body).simplify_trig().simplify_full() == 0), 'the antiderivative does not differentiate back'
value = (F.subs({X: I.upper}) - F.subs({X: I.lower}))
for _v in set(SR(I.lower).variables()) | set(SR(I.upper).variables()):
    assume(_v > 0); assume(_v < pi)  # a symbolic bound inside (0, π): Maxima evaluates the definite integral
assert goal.expr is not None and bool(SR(goal.expr).simplify_full().simplify_trig() == 0) or \
    bool(SR(goal.expr).simplify_full() == 0), f'the integral is {value.simplify_full()}, not the goal value'
# multiples k of sin(k y), cos(k y) in F, its derivative, the integrand and the value
def _mults(e):
    out = set()
    for f in e.operands() if e.operator() is not None else []:
        out |= _mults(f)
    op = e.operator()
    if op in (sin, cos):
        arg = e.operands()[0]
        coeffs = arg.coefficients(X) if arg.has(X) else []
        for v in [X] + [SR(s) for s in arg.variables()]:
            c = arg.coefficient(v)
            if c in ZZ and abs(c) >= 2:
                out.add(abs(ZZ(c)))
        if arg.is_numeric() or (arg / pi).is_numeric():
            pass
    return out
K = max(list(_mults(SR(F)) | _mults(SR(body)) | _mults(SR(diff(F, X)))) + [1])
cs, sn = var('c_ s_')
Cs, Ss = {1: cs}, {1: sn}
for k in range(2, K + 1):
    Cs[k] = (Cs[k - 1] * cs - Ss[k - 1] * sn).expand()
    Ss[k] = (Ss[k - 1] * cs + Cs[k - 1] * sn).expand()
def _poly_text(p, y):
    t = str(p).replace('c_', f'Real.cos {y}').replace('s_', f'Real.sin {y}').replace('^', ' ^ ').replace('*', ' * ')
    return t
facts, names = [], []
for k in range(2, K + 1):
    for kind, table, other in (('cos', Cs, Ss), ('sin', Ss, Cs)):
        name = f'{kind[0]}{k}'
        rhs = _poly_text(table[k], 'y')
        if k == 2:
            prf = (f'rw [show (2 : ℝ) * y = y + y by ring, Real.{kind}_add]\n    ring')
        else:
            prf = (f'rw [show ({k} : ℝ) * y = {k - 1} * y + y by ring, Real.{kind}_add, c{k - 1} y, s{k - 1} y]\n    ring')
        facts.append(f'have {name} : ∀ y : ℝ, Real.{kind} ({k} * y) = {rhs} := by\n    intro y\n    {prf}')
        names.append(name)
# the derivative of F, expanded, against the integrand, expanded: their difference over sin² + cos² - 1
def _expand(e):
    e = SR(e)
    for k in range(K, 1, -1):
        e = e.subs({cos(k * X): Cs[k].subs({cs: cos(X), sn: sin(X)}), sin(k * X): Ss[k].subs({cs: cos(X), sn: sin(X)})})
    return e.subs({cos(X): cs, sin(X): sn}).expand()
D = _expand(diff(F, X)) - _expand(body)
Rr = PolynomialRing(QQ, ['c_', 's_'] + [str(v) for v in D.variables() if str(v) not in ('c_', 's_')])
Dp = Rr(D)
pyth = Rr('c_')**2 + Rr('s_')**2 - 1
cof = Dp.quo_rem(pyth)[0] if Dp != 0 else Rr(0)
assert Dp == cof * pyth, 'the expanded derivative differs from the integrand beyond sin² + cos² = 1'
v = str(symbolic(I.var, problem, lean_type='')).strip('()')
cof_text = _poly_text(SR(cof), v)
F_lean = str(symbolic(F.subs({X: I.var}), problem, lean_type=''))
term = has_deriv_at(F.subs({X: I.var}), I.var, I.var, problem)
expand = ', '.join(names)
deriv_close = ('first | ring1 | (field_simp; done) | (field_simp; ring1) | '
               f'(simp only [{expand + ", " if expand else ""}one_mul, mul_one]; first | ring1 | '
               f'linear_combination ({cof_text}) * Real.cos_sq_add_sin_sq {v})')
lines = facts + [
    f'have hF : ∀ {v} ∈ Set.uIcc ({I.lower_text} : ℝ) ({I.upper_text} : ℝ), HasDerivAt (fun {v} => ({F_lean} : ℝ)) (({I.function}) {v}) {v} := by',
    f'  intro {v} _',
    f'  refine (({term}).congr_of_eventuallyEq (Filter.Eventually.of_forall fun y => ?_)).congr_deriv ?_',
    '  · beta_reduce; first | rfl | ring1 | (simp; ring1) | (simp; done) | (simp only [Pi.add_apply, Pi.sub_apply, Pi.mul_apply, Pi.neg_apply, Pi.pow_apply]; ring1)',
    f'  · beta_reduce\n    {deriv_close}',
    f'rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hF (by apply Continuous.intervalIntegrable; fun_prop)]',
    'beta_reduce',
    f'simp only [{expand + ", " if expand else ""}one_mul, mul_one, Real.sin_pi, Real.cos_pi, Real.sin_zero, Real.cos_zero, '
    'Real.sin_two_pi, Real.cos_two_pi, mul_zero, zero_mul, sub_zero, zero_sub, neg_zero, add_zero, zero_add]',
    'first | ring1 | (field_simp; done) | (field_simp; ring1) | (norm_num; done) | (field_simp; ring_nf; done) | (ring_nf; done)',
]
proof = tactic('\n'.join(lines))
