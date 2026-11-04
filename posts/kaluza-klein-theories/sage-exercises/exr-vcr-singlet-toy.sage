# Exercise exr-vcr-singlet-toy: a two-field toy model of a consistent truncation. s is a singlet and t changes sign
# under a Z2 symmetry. For the most general invariant quartic potential the t-equation is solved by t = 0 for every s;
# a non-invariant cubic coupling g s^2 t sources t, and setting t = 0 is no longer consistent.
s, t = var("s t")
a, b, c, d, e, g = var("a b c d e g")
V_inv = a * s ^ 2 + b * s ^ 4 + c * t ^ 2 + d * s ^ 2 * t ^ 2 + e * t ^ 4 + var("q") * s ^ 3
print("invariant potential: dV/dt at t = 0 is", diff(V_inv, t).subs(t=0))
V_bad = V_inv + g * s ^ 2 * t
print("with g s^2 t: dV/dt at t = 0 is", diff(V_bad, t).subs(t=0), "(a source for t)")
# circle version: modes phi_n with charge n; a cubic coupling phi_a phi_b phi_c is U(1)-invariant only if a + b + c = 0
inv = [(a_, b_, c_) for a_ in range(-2, 3) for b_ in range(a_, 3) for c_ in range(b_, 3) if a_ + b_ + c_ == 0]
print("U(1)-invariant cubic couplings among modes -2..2:", inv)
print("couplings linear in a non-zero mode with the other two zero modes:", [x for x in inv if sorted(x).count(0) == 2])
