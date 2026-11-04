# Exercise exr-qm-pt-two-level-sage: perturbative expansion of the exact eigenvalues of a two-level Hamiltonian.
E1, E2, v = var("E1 E2 v", domain="real"); assume(E2 > E1)
d = E2 - E1
low = (E1 + E2) / 2 - sqrt((d / 2) ** 2 + v ** 2); high = (E1 + E2) / 2 + sqrt((d / 2) ** 2 + v ** 2)
print("lower:", taylor(low, v, 0, 4).simplify_full()); print("upper:", taylor(high, v, 0, 4).simplify_full())
print("lower - (E1 - v^2/d + v^4/d^3) to order 4:", (taylor(low, v, 0, 4) - (E1 - v ** 2 / d + v ** 4 / d ** 3)).simplify_full())
ex = low.subs(E1=0, E2=1, v=0.2).n(); so = -0.2 ** 2
print("E1 = 0, E2 = 1, v = 0.2: exact", ex, " second order", so, " error", abs(ex - so), " v^4 =", 0.2 ** 4)
