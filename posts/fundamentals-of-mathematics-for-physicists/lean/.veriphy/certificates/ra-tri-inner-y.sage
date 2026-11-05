# name: ra-tri-inner-y
# technique: fundamental theorem of calculus with an antiderivative from Sage
# goal: ∫₀ˣ e^(x+y) dy = e^(2x) - eˣ
# description: Sage finds F with F′ = f and checks F b - F a = v; Lean proves HasDerivAt F f and evaluates the bounds.
INDEX = 0  # EDIT: which interval integral of the goal, when it has several
proof = definite_integral_proof(problem, INDEX)
from veriphy.toolkit._integral import DERIVATIVE_CLOSE, FUNCTION_CLOSE, VALUE_CLOSE
text = str(proof)
# the refine may leave one goal or two (the function equality is sometimes solved by unification): close whatever remains;
# ring_nf also normalises the argument order inside exp; the value step may need ring after norm_num
text = text.replace(f'  · {FUNCTION_CLOSE}\n  · {DERIVATIVE_CLOSE}', f'  all_goals first | ({FUNCTION_CLOSE}) | ({DERIVATIVE_CLOSE}) | (ring_nf; done) | (beta_reduce; ring_nf; done)')
text = text.replace('\nbeta_reduce\n', '\ntry beta_reduce\n')
text = text.replace(VALUE_CLOSE, VALUE_CLOSE + ' | (norm_num; ring1) | (simp; ring1) | (norm_num; ring_nf; done)')
proof = tactic(text)
