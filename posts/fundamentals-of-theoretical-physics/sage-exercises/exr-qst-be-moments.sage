# Exercise exr-qst-be-moments: for one bosonic mode with y = e^{-beta(eps - mu)} < 1, P(n) = (1 - y) y^n; check the
# mean <n> = y/(1 - y) = 1/(e^{beta(eps - mu)} - 1) and the variance <n>(1 + <n>); for a fermionic mode, n(1 - n).
n, y = var("n y"); assume(y > 0, y < 1)
P = (1 - y) * y ** n
m0 = SR(sum(P, n, 0, oo)).simplify_full(); m1 = SR(sum(n * P, n, 0, oo)).simplify_full(); m2 = SR(sum(n ** 2 * P, n, 0, oo)).simplify_full()
print("sum P =", m0, "; <n> =", m1, "; Var n =", (m2 - m1 ** 2).simplify_full(), "; <n>(1 + <n>) =", (m1 * (1 + m1)).simplify_full())
nF = y / (1 + y)    # fermions: P(0) = 1/(1 + y), P(1) = y/(1 + y)
print("fermions: <n> =", nF, "; Var n = <n^2> - <n>^2 =", (nF - nF ** 2).simplify_full(), "; <n>(1 - <n>) =", (nF * (1 - nF)).simplify_full())
