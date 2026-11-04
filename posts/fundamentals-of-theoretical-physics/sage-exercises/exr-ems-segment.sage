# Exercise exr-ems-segment (Pierrus, Question 2.9): field of a rod of length L, charge lambda per length, on its bisecting plane.
lam, L, s, zp = var('lam L s zp', domain='positive')
Es = (lam / (4 * pi) * integrate(s / (s^2 + zp^2)^(3/2), zp, -L / 2, L / 2)).simplify_full()
print("E_s =", Es)
print("equals lam L/(4 pi s sqrt(s^2 + L^2/4)):", bool((Es - lam * L / (4 * pi * s * sqrt(s^2 + L^2 / 4))).simplify_full() == 0))
print("limit L -> oo:", limit(Es, L=oo))
print("leading term for L << s:", Es.taylor(L, 0, 1))
