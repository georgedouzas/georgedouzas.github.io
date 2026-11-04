# Exercise exr-clm-circular-orbit: circular orbits, radial frequency U''(r0)/m and angular frequency L/(m r0^2)
# for the Kepler potential -k/r and the isotropic oscillator k r^2/2.
r, m, k, L = var("r m k L", domain="positive")
for name, V in [("Kepler -k/r", -k / r), ("oscillator k r^2/2", k * r**2 / 2)]:
    U = L**2 / (2 * m * r**2) + V
    r0 = [s.rhs() for s in solve(diff(U, r) == 0, r) if bool(s.rhs().subs(m=1, k=1, L=1) == 1)][0]
    wr2 = (diff(U, r, 2).subs(r=r0) / m).simplify_full()
    wphi2 = ((L / (m * r0**2))**2).simplify_full()
    print(name, ": r0 =", r0, "; omega_r^2 =", wr2, "; omega_phi^2 =", wphi2, "; ratio omega_r/omega_phi =", sqrt(wr2 / wphi2).simplify_full())
