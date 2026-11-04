# Exercise exr-ens-poisson: in the grand-canonical classical ideal gas, Xi = exp(z Z_1), so P(N) = e^{-<N>} <N>^N/N!.
# Check normalisation, mean and variance, and the ideal-gas law p V = T log Xi = <N> T.
N, lam, z, Z1, T = var("N lam z Z_1 T"); assume(lam > 0, z > 0, Z1 > 0, T > 0)
Xi = sum((z * Z1) ** N / factorial(N), N, 0, oo)
print("Xi = sum_N z^N Z_1^N/N! =", Xi)
Nmean = (z * diff(log(Xi), z)).simplify_full()
print("<N> = z d log Xi/dz =", Nmean, "; p V = T log Xi =", T * log(Xi).simplify_log())
P = exp(-lam) * lam ** N / factorial(N)
s0 = sum(P, N, 0, oo); s1 = sum(N * P, N, 0, oo); s2 = sum(N ** 2 * P, N, 0, oo)
print("sum P =", SR(s0).simplify_full(), "; <N> =", SR(s1).simplify_full(), "; Var N =", SR(s2 - s1 ** 2).simplify_full())
