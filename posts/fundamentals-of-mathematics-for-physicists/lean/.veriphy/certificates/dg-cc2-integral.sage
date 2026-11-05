# name: dg-cc2-integral
# technique: fundamental theorem of calculus with an antiderivative from Sage
# goal: the interval integrals of the Differential geometry chapter, part cc2: total curvatures and areas of the
#   sphere, the torus and their caps and halves (integrands 2 pi sin, 2 pi cos, 2 pi R0^2 sin, 2 pi (2 + cos a), ...)
# description: Sage finds F with F' = f and checks F b - F a = v; Lean proves HasDerivAt F f and evaluates the bounds.
INDEX = 0
proof = definite_integral_proof(problem, INDEX)
