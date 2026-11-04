# Exercise exr-qz-longitudinal-growth: the longitudinal polarisation eps_L = (|k|, 0, 0, E)/m of a massive vector with
# k = (E, 0, 0, |k|): eps_L . k = 0, eps_L . eps_L = -1, and eps_L - k/m = O(m/E).
E, m = var("E m", domain="positive")
kk = sqrt(E ** 2 - m ** 2)
eta = diagonal_matrix([1, -1, -1, -1])
k = vector([E, 0, 0, kk]); eps = vector([kk, 0, 0, E]) / m
print("eps_L . k =", (eps * eta * k).simplify_full(), "   eps_L . eps_L =", (eps * eta * eps).simplify_full())
diff0 = (eps - k / m)[0]
x = var("x")       # x = m/E
print("(eps_L - k/m)^0 as a series in m/E:", (diff0.subs(m=x * E)).simplify_full().series(x, 4))
