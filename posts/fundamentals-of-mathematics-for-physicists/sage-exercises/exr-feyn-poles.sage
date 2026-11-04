# Exercise exr-feyn-poles: the k0 integral of the Feynman propagator -e^(-i k0 t)/(k0^2 - omega^2 + i eps) by residues:
# pole positions for small eps, the enclosed pole for each sign of t, and the result i e^(-i omega |t|)/(2 omega).
k0, t, w, eps = var("k0 t omega eps")
assume(w > 0, eps > 0)
poles = solve(k0 ** 2 - w ** 2 + I * eps == 0, k0)
print("poles:", [p.rhs() for p in poles])
print("to first order in eps:", [p.rhs().taylor(eps, 0, 1) for p in poles])
integrand = -exp(-I * k0 * t) / ((k0 - w) * (k0 + w))         # eps -> 0 in the residues, kept in the pole positions
res_plus = (integrand * (k0 - w)).subs(k0=w)                   # residue at k0 = omega (below the axis)
res_minus = (integrand * (k0 + w)).subs(k0=-w)                 # residue at k0 = -omega (above the axis)
result_tpos = (-2 * pi * I * res_plus / (2 * pi)).simplify_full()    # t > 0: close below, clockwise
result_tneg = (2 * pi * I * res_minus / (2 * pi)).simplify_full()    # t < 0: close above, counter-clockwise
print("t > 0:", result_tpos, "| t < 0:", result_tneg)
print("both equal i e^(-i omega |t|)/(2 omega):", bool((result_tpos - I * exp(-I * w * t) / (2 * w)).simplify_full() == 0), bool((result_tneg - I * exp(I * w * t) / (2 * w)).simplify_full() == 0))
