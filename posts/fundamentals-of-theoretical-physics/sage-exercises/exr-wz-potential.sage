# Exercise exr-wz-potential: eliminate the auxiliary field and read off the potential and masses of the Wess-Zumino
# model with W = m phi^2/2 + y phi^3/6.
m, y = var("m y"); assume(m > 0, y > 0)
phi, phis, F, Fs = var("phi phis F Fs")
W = m * phi^2 / 2 + y * phi^3 / 6; Wp = diff(W, phi); Wps = Wp.subs(phi=phis)   # W' and its conjugate (phi* independent)
Laux = Fs * F + Wp * F + Wps * Fs                                                # Martin eq. 3.2.16 context
Fsol = solve([diff(Laux, Fs) == 0, diff(Laux, F) == 0], [F, Fs], solution_dict=True)[0]
print("auxiliary field equations: F =", Fsol[F], ", F* =", Fsol[Fs])
V = -Laux.subs(Fsol).expand()
print("scalar potential V = -L_aux on the solution =", V.factor())
print("[PASS] F = -W'^* and V = |W'|^2 = |m phi + y phi^2/2|^2:", bool(Fsol[F] == -Wps) and bool((V - Wp * Wps).expand() == 0))
print("[PASS] V vanishes at phi = 0 and phi = -2m/y, the supersymmetric vacua:", bool(V.subs(phi=0, phis=0) == 0) and bool(V.subs(phi=-2 * m / y, phis=-2 * m / y) == 0))
# masses at phi = 0: scalar mass^2 from the |phi|^2 coefficient, fermion mass from W''(0) = m
print("[PASS] the scalar mass squared (coefficient of |phi|^2 in V) equals the fermion mass squared |W''(0)|^2 = m^2:",
      bool(V.coefficient(phi * phis) == m^2) and bool(diff(W, phi, 2).subs(phi=0) == m))
print("Yukawa coupling W'''(phi) = y, the same y that appears in the quartic |y phi^2/2|^2 term of V:", diff(W, phi, 3))
