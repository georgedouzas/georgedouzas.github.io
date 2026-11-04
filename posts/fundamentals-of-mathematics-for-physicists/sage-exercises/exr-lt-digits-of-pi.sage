# Exercise exr-lt-digits-of-pi: frequencies of the digits 0..9 among the first 10^5 decimal digits of pi, and their
# deviations from 10^4 in units of sqrt(10^5 * 0.1 * 0.9) = 94.9, the standard deviation for independent uniform digits.
N = 10 ** 5
digits = str(pi.n(digits=N + 10))[2:N + 2]
print("number of digits used:", len(digits))
sd = sqrt(N * 0.1 * 0.9)
counts = [digits.count(str(d)) for d in range(10)]
for d, c in enumerate(counts):
    print("digit %d: count %5d, deviation %+4d = %+.2f standard deviations" % (d, c, c - N // 10, (c - N // 10) / sd))
print("chi-squared statistic sum (c - 10^4)^2 / 10^4 = %.2f (9 degrees of freedom, mean 9)" % sum((c - N / 10) ** 2 / (N / 10) for c in counts))
