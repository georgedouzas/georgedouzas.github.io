# Exercise exr-ems-rods: force between two collinear rods of length L with gap a, charge lambda per length (Heaviside-Lorentz).
lam, L, a, u, v = var('lam L a u v', domain='positive')
F = lam^2 / (4 * pi) * integrate(integrate(1 / (a + u + v)^2, v, 0, L), u, 0, L)
F = F.simplify_log()
print("F =", F)
print("equals lam^2/(4 pi) log((a+L)^2/(a(a+2L))):", bool((F - lam^2 / (4 * pi) * log((a + L)^2 / (a * (a + 2 * L)))).simplify_log().simplify_full() == 0))
print("expansion for small L:", F.taylor(L, 0, 3))
