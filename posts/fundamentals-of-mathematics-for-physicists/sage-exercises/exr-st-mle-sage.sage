# Exercise exr-st-mle-sage: maximum likelihood with symbolic data: the Gaussian log-likelihood of x1..x4 is stationary
# at mu = sample mean and v = biased sample variance; the Poisson log-likelihood of counts k1..k4 at lambda = mean count.
m, v, lam = var("m v lam")
assume(v > 0); assume(lam > 0)
xs = var("x1 x2 x3 x4")
ll = sum(-(xi - m) ** 2 / (2 * v) - log(2 * pi * v) / 2 for xi in xs)
sol = solve([diff(ll, m) == 0, diff(ll, v) == 0], [m, v], solution_dict=True)
xbar = sum(xs) / 4
vb = sum((xi - xbar) ** 2 for xi in xs) / 4
for s in sol:
    print("Gaussian: mu_hat =", s[m], "; equals the sample mean:", bool((s[m] - xbar).expand() == 0))
    print("          sigma_hat^2 =", s[v].expand(), "; equals the biased sample variance:", bool((s[v] - vb).expand() == 0))
ks = var("k1 k2 k3 k4")
assume(k1 > 0); assume(k2 > 0); assume(k3 > 0); assume(k4 > 0)
llp = sum(ki * log(lam) - lam - log(factorial(ki)) for ki in ks)
solp = solve(diff(llp, lam) == 0, lam)
print("Poisson: lambda_hat =", solp[0].rhs(), "; second derivative there:", diff(llp, lam, 2).subs(lam=solp[0].rhs()).simplify_full(), "< 0 for positive counts")
