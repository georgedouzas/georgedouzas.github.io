# Exercise exr-pbr-large-deviations (Bricmont, Sec. 2.3 and Exercise 2.10): a coin with P(heads) = 1/3 tossed N times;
# the exact probability that the frequency of heads lies within eps = 0.01 of 1/2, against the Chernoff bound
# exp(-N I(1/2 - eps)) with the relative entropy I(a) = a ln(3a) + (1 - a) ln(3(1 - a)/2), and the decay rate
# -ln(P)/N against I(1/2) = ln(9/8)/2.
p = 1/3; eps = 1/100
I = lambda a: a*log(a/p) + (1 - a)*log((1 - a)/(1 - p))
print("I(1/2) = ln(9/8)/2 = %.6f, I(1/2 - eps) = %.6f" % (I(1/2).n(), I(1/2 - eps).n()))
for N in [100, 1000, 10000]:
    P = sum(binomial(N, k)*p^k*(1 - p)^(N - k) for k in range(ceil(N*(1/2 - eps)), floor(N*(1/2 + eps)) + 1))
    print("N = %5d: P = %.4e, Chernoff bound %.4e, -ln(P)/N = %.6f" % (N, P.n(), exp(-N*I(1/2 - eps)).n(), (-log(P)/N).n()))
print("fair coin, same window: P = %.4f (N = 10000), tending to 1" % sum(binomial(10000, k)/2^10000 for k in range(4900, 5101)).n())
