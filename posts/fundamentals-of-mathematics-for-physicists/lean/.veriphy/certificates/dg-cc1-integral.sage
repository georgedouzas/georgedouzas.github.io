# name: dg-cc1-integral
# technique: fundamental theorem of calculus with an antiderivative from Sage
# goal: ∫ x in a..b, f x = v, with f elementary (polynomials, exp, log, sin, cos, sqrt, powers)
# description: Sage finds F with F' = f and checks F b - F a = v; Lean proves HasDerivAt F f and evaluates the bounds.
INDEX = 0  # EDIT: which interval integral of the goal, when it has several
proof = definite_integral_proof(problem, INDEX)
