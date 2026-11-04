# Exercise exr-ratio-sage: the ratio test. The limits of a_{n+1}/a_n for a_n = 2^n/n!, n!/n^n, n^2/2^n and 1/n decide
# convergence when they differ from 1; Sage confirms the sums of the convergent ones.
n = var("n")
for a, name in [(2 ** n / factorial(n), "2^n/n!"), (factorial(n) / n ** n, "n!/n^n"), (n ** 2 / 2 ** n, "n^2/2^n"), (1 / n, "1/n")]:
    L = limit(a.subs(n=n + 1) / a, n=oo)
    verdict = "converges" if bool(L < 1) else ("diverges" if bool(L > 1) else "inconclusive")
    print("a_n =", name, ": lim a_{n+1}/a_n =", L, "->", verdict)
S2 = sum(2 ** n / factorial(n), n, 1, oo)
print("sum 2^n/n! =", S2, "= e^2 - 1 (2 e sinh 1 = e^2 - 1):", bool(abs(numerical_approx(S2 - (exp(2) - 1), digits=30)) < 1e-25), "; sum n^2/2^n =", sum(n ** 2 / 2 ** n, n, 1, oo))
