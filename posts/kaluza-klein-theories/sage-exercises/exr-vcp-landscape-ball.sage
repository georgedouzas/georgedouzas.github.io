# Exercise exr-vcp-landscape-ball: the number of integer flux vectors n in Z^d with |n|^2 <= L against the volume
# pi^{d/2} L^{d/2}/Gamma(d/2 + 1) of the ball, for d = 1, ..., 5 and L = 400 (radius 20): the ratio tends to 1 as L
# grows, which is the continuum approximation behind the Bousso-Polchinski estimate.
def count(d, L):
    # number of lattice points in the ball, via the theta-series coefficients of Z^d
    th = [0] * (L + 1)
    k = 0
    while k * k <= L:
        th[k * k] += 1 if k == 0 else 2
        k += 1
    tot = [1] + [0] * L
    for _ in range(d):
        new = [0] * (L + 1)
        for i, ci in enumerate(tot):
            if ci:
                for j in range(0, L + 1 - i):
                    if th[j]:
                        new[i + j] += ci * th[j]
        tot = new
    return sum(tot)
L = 400
for d in range(1, 6):
    c = count(d, L); v = (pi**(d / 2) * L**(d / 2) / gamma(d / 2 + 1)).n()
    print("d =", d, ": lattice points", c, "; ball volume", v.n(digits=7), "; ratio", (c / v).n(digits=4))
print("Bousso-Polchinski: d = 100, L = 100 gives pi^50 10^100/50! =", (pi**50 * 10**100 / factorial(50)).n(digits=3))
