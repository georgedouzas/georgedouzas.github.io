# Exercise exr-vcp-nk-torsion: the torsion class W_2 of SU(3)/(U(1) x U(1)) with radii R1, R2, R3 (Chatzistavrakidis-
# Zoupanos, eq. B.31) has the three coefficients R_i^2 (2 R_i^2 - R_j^2 - R_k^2); it vanishes, so the coset is
# nearly-Kahler, exactly when R1 = R2 = R3, and then W_1 = -(2i/3)(R1^2 + R2^2 + R3^2)/(R1 R2 R3) = -2i/R. For
# Sp(4)/(SU(2) x U(1)) (eq. B.20) W_2 vanishes exactly for R1 = R2, and W_1 = -(2i/3)(2R1^2 + R2^2)/(R1^2 R2) -> -2i/R.
R1, R2, R3, Rr = var("R1 R2 R3 R", domain="positive")
c = [R1**2 * (2 * R1**2 - R2**2 - R3**2), R2**2 * (2 * R2**2 - R1**2 - R3**2), R3**2 * (2 * R3**2 - R1**2 - R2**2)]
sol = solve([x == 0 for x in c], R2, R3)
print("SU(3)/U(1)^2: solutions of W_2 = 0 with positive radii:", [s for s in sol if all(bool(e.rhs() > 0) for e in s)] if sol else sol)
W1 = -2 * I / 3 * (R1**2 + R2**2 + R3**2) / (R1 * R2 * R3)
print("W_1 at R1 = R2 = R3 = R:", W1.subs(R1=Rr, R2=Rr, R3=Rr).simplify_full())
print("W_2 coefficients at R1 = R2 = R3:", [x.subs(R2=R1, R3=R1) for x in c])
print("W_2 at R1 = 1, R2 = 1, R3 = 2 (not nearly-Kahler):", [x.subs(R1=1, R2=1, R3=2) for x in c])
csp = [R1**2 * (R1**2 - R2**2), -2 * R2**2 * (R2**2 - R1**2)]
print("Sp(4)/(SU(2) x U(1)): W_2 coefficients", csp, "vanish iff R1 = R2:", bool(all(x.subs(R2=R1) == 0 for x in csp)), "; W_1 at R1 = R2 = R:",
      (-2 * I / 3 * (2 * R1**2 + R2**2) / (R1**2 * R2)).subs(R1=Rr, R2=Rr).simplify_full())
