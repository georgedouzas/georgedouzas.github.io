# Exercise exr-ra-wallis (integrative): the Wallis integrals I_n = int_0^{pi/2} sin^n x dx from the reduction formula
# n I_n = (n - 1) I_{n-2}, the squeeze I_{2n+1} <= I_{2n} <= I_{2n-1}, and the Wallis product for pi/2 as the limit
# of I_{2n}/I_{2n+1} -> 1.
x = var("x")
I = {0: pi / 2, 1: 1}
for n in range(2, 15):
    I[n] = (n - 1) * I[n - 2] / n
print("I_n from the reduction formula agrees with direct integration for n <= 6:", all(bool(I[n] == integrate(sin(x) ** n, x, 0, pi / 2)) for n in range(7)))
print("I_4 =", I[4], ", I_5 =", I[5])
for n in [1, 3, 6]:
    print("n =", n, ": I_{2n+1} <= I_{2n} <= I_{2n-1}:", bool(I[2 * n + 1] <= I[2 * n] <= I[2 * n - 1]), "; I_{2n}/I_{2n+1} =", (I[2 * n] / I[2 * n + 1]).n(digits=8))
W = lambda m: prod(QQ(4 * k ** 2) / (4 * k ** 2 - 1) for k in range(1, m + 1))
print("Wallis product prod_{k<=m} 4k^2/(4k^2-1) for m = 10, 100, 1000:", [W(m).n(digits=8) for m in [10, 100, 1000]], "-> pi/2 =", (pi / 2).n(digits=8))
