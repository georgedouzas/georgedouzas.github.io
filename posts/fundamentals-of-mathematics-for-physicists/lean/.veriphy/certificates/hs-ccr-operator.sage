# name: hs-ccr-operator
# technique: operator identity on a test function (operator_identity recipe): [d/dx, x] ψ = ψ
# goal: deriv (fun y => y * ψ y) x - x * deriv ψ x = ψ x, for ψ continuously differentiable
# description: Sage computes the commutator of d/dx and multiplication by x on a symbolic ψ; Lean proves it with the
#   derivative lemmas for products.
COORDINATES = Coordinates(1)
LHS = commutator(COORDINATES.partial(0), COORDINATES.multiplication(COORDINATES.symbols[0]))
RHS = COORDINATES.scalar(1)
CLOSE = 'simpa using hop'
facts = operator_identity(LHS, RHS, COORDINATES)
proof = steps(facts, tactic(CLOSE))
