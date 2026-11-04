# Exercise exr-pfn-geometric: the oscillator partition function as a geometric series, its mean energy, and the
# high-temperature expansion U = T + omega^2/(12 T) - omega^4/(720 T^3) + ... (hbar = k_B = 1).
n, beta, w, T = var("n beta omega T"); assume(beta > 0, w > 0, T > 0)
Z = sum(exp(-beta * w * (n + 1 / 2)), n, 0, oo)
print("Z =", Z.simplify_full())
U = (-diff(log(Z), beta)).simplify_full()
print("U =", U, "; U - (omega/2 + omega/(e^{beta omega} - 1)) =", (U - (w / 2 + w / (exp(beta * w) - 1))).simplify_full())
UT = U.subs(beta=1 / T)
eps_ = var("eps_")      # expand in eps_ = 1/T
print("high-T expansion: U =", (UT.subs(T=1 / eps_)).taylor(eps_, 0, 3).subs(eps_=1 / T))
print("mean occupation <n> = (U - omega/2)/omega =", ((U - w / 2) / w).simplify_full())
