# Exercise exr-qm-rabi-sage: U(t) = exp(-i H t/hbar) for H = (hbar Omega/2) sigma_x and the spin-flip probability.
t, Om = var("t Om", domain="positive"); hbar = var("hbar", domain="positive")
sx = matrix(SR, [[0, 1], [1, 0]])
U = (-I * (hbar * Om / 2) * sx * t / hbar).exp().apply_map(lambda e: e.exponentialize().simplify_full().rectform().trig_reduce())
print("U(t) ="); print(U)
amp = U[1, 0]
P = (amp * amp.conjugate()).simplify_full()
print("Prob(down at t) =", P, "; equals sin(Om t/2)^2:", bool((P - sin(Om * t / 2) ** 2).trig_reduce() == 0))
print("first t with probability 1:", solve(Om * t / 2 == pi / 2, t))
