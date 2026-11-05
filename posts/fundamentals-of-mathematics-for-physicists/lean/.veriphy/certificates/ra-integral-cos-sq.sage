# name: ra-integral-cos-sq
# technique: fundamental theorem of calculus with an antiderivative from Sage
# goal: ∫ cos² θ over [-π/2, π/2] = π/2
# description: Sage finds F with F′ = f and checks F b - F a = v; Lean proves HasDerivAt F f and evaluates the bounds.
INDEX = 0  # EDIT: which interval integral of the goal, when it has several
from veriphy.toolkit._integral import VALUE_CLOSE
text = str(definite_integral_proof(problem, INDEX))
assert VALUE_CLOSE in text and 'he0' in text
# the lower bound -π/2 gives sin(2·(-π/2)) = -sin(2·(π/2)), so the endpoint fact must be used after the sign is pulled out
proof = tactic(text.replace(VALUE_CLOSE, 'all_goals first | (norm_num [he0]; done) | (norm_num [he0]; ring1) | (simp [he0]; ring1) | (norm_num [he0, Real.sin_pi]; done) | ' + VALUE_CLOSE[len('all_goals first | '):]))
