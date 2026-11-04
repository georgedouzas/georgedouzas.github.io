# Exercise exr-ren-2l-chain: the massless bubble in the modified minimal scheme, 16 pi^2 mu^eps B = e^(gamma eps/2)
# (4 pi)^(d/2) G(1,1) (p^2/mu^2)^(-eps/2), d = 4 - eps, L = log(p^2/mu^2), and the chain with its counterterm graphs.
var('e L'); d = 4 - e
G = lambda a, b: gamma(a + b - d / 2) * gamma(d / 2 - a) * gamma(d / 2 - b) / (gamma(a) * gamma(b) * gamma(d - a - b))
B = exp(euler_gamma * e / 2) * G(1, 1) * exp(-e * L / 2)
print("16 pi^2 B =", B.series(e, 1).truncate().expand())
chain = B^2
print("chain poles: 1/e^2:", chain.series(e, 1).truncate().expand().coefficient(e, -2), "  1/e:", chain.series(e, 1).truncate().expand().coefficient(e, -1))
sub = (B^2 - 2 * (2 / e) * B).series(e, 1).truncate().expand()
print("chain minus counterterm graphs: 1/e^2:", sub.coefficient(e, -2), "  1/e:", sub.coefficient(e, -1))
