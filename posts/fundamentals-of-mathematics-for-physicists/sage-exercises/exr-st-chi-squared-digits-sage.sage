# Exercise exr-st-chi-squared-digits-sage: chi-squared test of the digit counts of the first 10^5 decimals of pi
# against uniformity: the statistic, its p-value with 9 degrees of freedom from RealDistribution and from the
# incomplete Gamma function, and the 5 and 95 percent quantiles of chi^2_9 and chi^2_5.
N = 10 ** 5
digits = str(pi.n(digits=N + 10))[2:N + 2]
counts = [digits.count(str(d)) for d in range(10)]
chi2 = sum((QQ(c) - N / 10) ** 2 / (N / 10) for c in counts)
print("counts:", counts)
print("chi^2 =", chi2, "=", numerical_approx(chi2, digits=4))
C9 = RealDistribution("chisquared", 9)
p_rd = 1 - C9.cum_distribution_function(float(chi2))
p_gamma = float(gamma_inc(QQ(9) / 2, chi2 / 2) / gamma(QQ(9) / 2))   # upper incomplete Gamma, normalised
print("p-value (RealDistribution): %.4f ; p-value (incomplete Gamma): %.4f" % (p_rd, p_gamma))
for k in [9, 5]:
    C = RealDistribution("chisquared", k)
    print("chi^2_%d quantiles: 5%% = %.3f, 95%% = %.3f" % (k, C.cum_distribution_function_inv(0.05), C.cum_distribution_function_inv(0.95)))
