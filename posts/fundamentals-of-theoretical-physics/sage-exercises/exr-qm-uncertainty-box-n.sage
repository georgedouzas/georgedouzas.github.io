# Exercise exr-qm-uncertainty-box-n: Delta x, Delta p and their product in the box eigenstates n = 1..6.
x = var("x", domain="real"); hbar, L = var("hbar L", domain="positive")
for n in range(1, 7):
    ph = sqrt(2 / L) * sin(n * pi * x / L)
    E = lambda g: integrate(g, x, 0, L).simplify_full()
    vx = (E(x ** 2 * ph ** 2) - E(x * ph ** 2) ** 2).simplify_full(); vp = E(ph * (-hbar ** 2) * diff(ph, x, 2))
    prod = sqrt(vx * vp).simplify_full()
    print("n =", n, " (Delta x)^2 =", vx, " Delta p =", sqrt(vp).simplify_full(), " Delta x Delta p / hbar =", (prod / hbar).simplify_full(), "=", (prod / hbar).n(digits=5),
          " matches (1/2) sqrt(n^2 pi^2/3 - 2):", bool(((prod / hbar) ** 2 - (n ** 2 * pi ** 2 / 3 - 2) / 4).simplify_full() == 0))
print("limit of (Delta x/L)^2 as n -> oo:", limit(1 / 12 - 1 / (2 * var("n") ** 2 * pi ** 2), n=oo))
