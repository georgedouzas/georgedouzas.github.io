# name: fa-integral2
# technique: fundamental theorem of calculus with an antiderivative from Sage (definite_integral recipe), with the
#   derivative and value steps widened (field_simp alone, norm_num then ring1, simp then ring1)
# goal: ∫ x in a..b, f x = v, f elementary (exp, powers, products), the value a closed form in e and π
# description: Sage finds F with F' = f and checks F b - F a = v; Lean proves HasDerivAt F f and evaluates the bounds;
#   the closing tactics accept values such as 1 - 2 e^{-1} that need simp before ring.
from veriphy.toolkit._integral import VALUE_CLOSE, DERIVATIVE_CLOSE
INDEX = 0
proof = definite_integral_proof(problem, INDEX)
text = getattr(proof, 'text', str(proof))
text = text.replace(VALUE_CLOSE, 'all_goals first | (norm_num; done) | ring1 | (field_simp; ring1) | (simp; done) '
                    '| (norm_num; ring1) | (simp; ring1) | (field_simp; done) | (simp; field_simp; ring1)')
text = text.replace(DERIVATIVE_CLOSE, DERIVATIVE_CLOSE + ' | (field_simp; done) | (simp; field_simp; ring1)')
proof = tactic(text)
