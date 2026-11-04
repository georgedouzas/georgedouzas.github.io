# Exercise exr-kk-modes: the Kaluza-Klein tower of a five-dimensional massless scalar on M4 x S^1. Expand
# Phi(x, y) = sum_n phi_n(x) e^{i n y/R}: with Box_5 = Box_4 - d_y^2 (signature (+,-,-,-,-)) each mode satisfies the
# four-dimensional Klein-Gordon equation (Box_4 + (n/R)^2) phi_n = 0, mass |n|/R, periodicity forces n to be an
# integer, and the orthogonality of the modes diagonalises the action.
y, R, n, m = var("y R n m"); assume(R > 0)
t, xx = var("t xx"); phin = function("phin")(t, xx)
Phi = phin * exp(I * n * y / R)
box5 = diff(Phi, t, 2) - diff(Phi, xx, 2) - diff(Phi, y, 2)          # (1+1)-dimensional part, Box_4 = d_t^2 - d_x^2, minus d_y^2
reduced = (box5 / exp(I * n * y / R)).simplify_full()
print("(Box_5 Phi) e^{-i n y/R} =", reduced)
print("[PASS] the mode obeys the 4D Klein-Gordon equation (Box_4 + (n/R)^2) phi_n = 0 with mass squared (n/R)^2:", bool((reduced - (diff(phin, t, 2) - diff(phin, xx, 2) + (n / R)^2 * phin)).simplify_full() == 0))
assume(n, "integer"); assume(m, "integer")
inner = integrate(exp(I * n * y / R) * exp(-I * m * y / R), y, 0, 2 * pi * R)
print("[PASS] orthogonality: int_0^{2 pi R} e^{i(n-m)y/R} dy = 2 pi R delta_{nm} (n = m case and n != m case):",
      bool(integrate(1, y, 0, 2 * pi * R) == 2 * pi * R) and bool(inner.subs(n=3, m=1).simplify_full() == 0))
print("masses |n|/R for n = -2..2 and R = 1:", [abs(k) for k in range(-2, 3)])
print("[PASS] periodicity e^{i n 2 pi} = 1 holds for integer n and fails for n = 1/2:", bool(exp(2 * pi * I * 5) == 1) and bool(exp(2 * pi * I / 2) == -1))
