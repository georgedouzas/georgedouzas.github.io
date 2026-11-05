# name: ca-spec-integral
# technique: fundamental theorem of calculus with an antiderivative from Sage (definite_integral recipe)
# goal: ∫ x in a..b, f x = v, with f a polynomial or a power of sin, bounds rationals or multiples of π
# description: Sage finds F with F' = f and checks F b - F a = v; Lean proves HasDerivAt F f and evaluates the bounds.
INDEX = 0
proof = definite_integral_proof(problem, INDEX)
