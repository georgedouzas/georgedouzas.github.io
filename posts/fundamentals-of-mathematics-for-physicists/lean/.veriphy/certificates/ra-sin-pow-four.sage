# name: ra-sin-pow-four
# technique: fundamental theorem of calculus with an antiderivative from Sage
# goal: ∫₀^(π/2) sin⁴ x dx = 3π/16
# description: Sage finds F with F′ = f and checks F b - F a = v; Lean proves HasDerivAt F f and evaluates the bounds.
INDEX = 0  # EDIT: which interval integral of the goal, when it has several
from veriphy.toolkit._integral import DERIVATIVE_CLOSE
proof = definite_integral_proof(problem, INDEX)

# the antiderivative's derivative equals the integrand only modulo sin² + cos² = 1 (and the double-angle formulas):
# Sage computes the cofactor q with F' - f = q (sin² + cos² - 1), after the same rewriting Lean performs
from sage.all import SR, var, sin, cos, PolynomialRing, QQ, integrate
from veriphy.toolkit._integral import VARIABLE
xv = SR.var(VARIABLE)
integral = problem.integrals[INDEX]
fbody = integral.body.subs({integral.var: xv})
Fanti = integrate(fbody, xv)
s_, c_ = var('s_ c_')
dexpr = (Fanti.diff(xv) - fbody).subs({cos(4 * xv): 2 * (2 * c_ ** 2 - 1) ** 2 - 1, sin(4 * xv): 2 * (2 * s_ * c_) * (2 * c_ ** 2 - 1),
                                        cos(2 * xv): 2 * c_ ** 2 - 1, sin(2 * xv): 2 * s_ * c_}).subs({sin(xv): s_, cos(xv): c_})
Rsc = PolynomialRing(QQ, 's_, c_')
q, rem = Rsc(dexpr).quo_rem(Rsc(s_ ** 2 + c_ ** 2 - 1))
assert rem == 0, f'the derivative check is not a multiple of sin² + cos² - 1: remainder {rem}'
qtext = str(q).replace('s_', f'Real.sin {VARIABLE[:-1]}').replace('c_', f'Real.cos {VARIABLE[:-1]}')
xl = VARIABLE[:-1]
TRIGCLOSE = (f"((try simp only [show (4 : ℝ) * {xl} = 2 * (2 * {xl}) by ring, Real.cos_two_mul, Real.sin_two_mul]); "
             f"linear_combination ({qtext}) * Real.sin_sq_add_cos_sq {xl})")
proof = tactic(str(proof).replace(DERIVATIVE_CLOSE, DERIVATIVE_CLOSE.replace('first | ', 'first | ' + TRIGCLOSE + ' | ', 1)))
