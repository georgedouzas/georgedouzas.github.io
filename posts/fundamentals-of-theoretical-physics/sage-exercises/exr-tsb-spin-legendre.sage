# Exercise exr-tsb-spin-legendre (Bricmont, Exercises 6.3-6.5): N variables s_i = +-1 with energy E = sum s_i (k = 1).
# Microcanonical entropy per spin s(e) from the binomial count, canonical free energy per spin f(T), and the
# Legendre relation f(T) = min_e (e - T s(e)).
e, T = var('e T'); assume(T > 0, e > -1, e < 1)
s = -((1 + e) / 2 * log((1 + e) / 2) + (1 - e) / 2 * log((1 - e) / 2))
f = -T * log(2 * cosh(1 / T))
for N in (100, 1000, 10000):
    E = N // 5
    exact = log(binomial(N, (N + E) // 2)).n() / N
    print("N = %5d, e = 0.2: log(Omega)/N = %.6f, s(e) = %.6f" % (N, exact, s.subs(e=0.2).n()))
print("s'(e) =", diff(s, e).simplify_full(), " = -artanh(e), so s'(e) = 1/T at e* = -tanh(1/T):",
      all(bool(abs(diff(s, e).subs(e=-tanh(1 / t)).n() - 1 / t) < 1e-12) for t in (0.3, 1, 5)))
val = (e - T * s).subs(e=-tanh(1 / T))
print("f(T) = e* - T s(e*) at e* = -tanh(1/T), numerically at T = 0.3, 1, 5:",
      [bool(abs(val.subs(T=t).n() - f.subs(T=t).n()) < 1e-12) for t in (0.3, 1, 5)])
print("and back: s(e) = min_T (e - f(T))/T at e = -0.4 gives", ((e - f) / T).subs(e=-0.4).find_local_minimum(0.05, 20)[0], "vs s(-0.4) =", s.subs(e=-0.4).n(digits=10))
