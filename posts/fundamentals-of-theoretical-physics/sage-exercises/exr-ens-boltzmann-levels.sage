# Exercise exr-ens-boltzmann-levels: Boltzmann probabilities of the levels E_n = n, n = 0, 1, 2, ..., at T = 1/2, 1, 3,
# the exact mean energy from the geometric series, and the truncation error of the figure's ten levels.
n, beta = var("n beta"); assume(beta > 0)
Z = sum(exp(-beta * n), n, 0, oo); U = (-diff(log(Z), beta)).simplify_full()
print("Z =", Z, "; <E> =", U, "= 1/(e^beta - 1):", bool((U - 1 / (exp(beta) - 1)).simplify_full() == 0))
for T in [1 / 2, 1, 3]:
    b = 1 / T
    p = [exp(-b * k) * (1 - exp(-b)) for k in range(4)]
    print("T = %s: p_0..p_3 =" % T, [x.n(digits=4) for x in p], "; <E> =", U.subs(beta=b).n(digits=5), "; weight beyond level 9 =", exp(-10 * b).n(digits=3))
