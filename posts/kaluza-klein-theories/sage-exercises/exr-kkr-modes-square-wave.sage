# Exercise exr-kkr-modes-square-wave: the Kaluza-Klein content of a field profile. The square wave Phi(y) = 1 on
# (0, pi R), -1 on (pi R, 2 pi R) has Fourier coefficients phi_n = 2/(i pi n) for odd n and 0 for even n: no zero mode,
# only odd massive modes, with weights decaying like 1/n; Parseval's identity (1/2piR) int |Phi|^2 = sum |phi_n|^2 = 1.
y, R = var("y R"); assume(R > 0)
def phin(n):
    return (integrate(exp(-I * n * y / R), y, 0, pi * R) - integrate(exp(-I * n * y / R), y, pi * R, 2 * pi * R)) / (2 * pi * R)
co = {n: phin(n).simplify_full() for n in range(-5, 6)}
print("phi_n for n = -5..5:", [co[n] for n in range(-5, 6)])
print("[PASS] phi_n = 2/(i pi n) for odd n and 0 for even n (no zero mode):", all(bool(co[n] == (2 / (I * pi * n) if n % 2 else 0)) for n in range(-5, 6)))
k = var("k")
total = 2 * sum(4 / (pi^2 * (2 * k + 1)^2), k, 0, oo)            # n = 2k + 1 > 0 and n = -(2k + 1)
print("[PASS] Parseval: sum over odd n of |2/(pi n)|^2 = 1 = (1/2 pi R) int |Phi|^2:", bool(total == 1))
