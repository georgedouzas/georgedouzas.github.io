# name: ra-ball-r
# technique: fundamental theorem of calculus with an antiderivative from Sage
# goal: ∫₀¹ r² sin θ dr = sin θ / 3
# description: Sage finds F with F′ = f and checks F b - F a = v; Lean proves HasDerivAt F f and evaluates the bounds.
INDEX = 0  # EDIT: which interval integral of the goal, when it has several
proof = definite_integral_proof(problem, INDEX)
