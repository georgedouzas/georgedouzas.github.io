# name: dg-ch-integral
# technique: fundamental theorem of calculus with an antiderivative from Sage (definite_integral recipe)
# goal: ∫ x in a..b, f x = v, with f elementary (polynomials, exp, log, sin, cos, sqrt, powers)
# description: Sage finds F with F' = f and checks F b - F a = v; Lean proves HasDerivAt F f and evaluates the bounds.
#   A symbolic bound is assumed inside (0, π) for Sage's evaluation of the definite integral (Maxima asks for its
#   sign); the Lean proof holds for every value of the bound.
INDEX = 0
I = problem.integrals[INDEX]
for _v in set(SR(I.lower).variables()) | set(SR(I.upper).variables()):
    assume(_v > 0); assume(_v < pi)
proof = definite_integral_proof(problem, INDEX)
