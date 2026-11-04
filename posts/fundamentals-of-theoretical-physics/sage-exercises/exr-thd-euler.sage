# Exercise exr-thd-euler: for the monatomic ideal gas U(S, V, N) = (3/2) N (N/V)^(2/3) exp(2S/(3N) - s0), compute
# T, p, mu, verify the Euler relation U = T S - p V + mu N, G = mu N and the Gibbs-Duhem relation.
S, V, N, s0 = var("S V N s0"); assume(V > 0, N > 0)
U = 3 / 2 * N * (N / V) ** (2 / 3) * exp(2 * S / (3 * N) - s0)
T = diff(U, S); p = -diff(U, V); mu = diff(U, N)
print("T =", T.simplify_full()); print("p V - N T =", (p * V - N * T).simplify_full())
print("mu =", mu.simplify_full())
print("U - (T S - p V + mu N) =", (U - (T * S - p * V + mu * N)).simplify_full())
G = U - T * S + p * V
print("G - mu N =", (G - mu * N).simplify_full())
# Gibbs-Duhem: S dT - V dp + N dmu = 0 along any variation of (S, V, N)
for v in [S, V, N]:
    print("Gibbs-Duhem along d" + str(v) + ":", (S * diff(T, v) - V * diff(p, v) + N * diff(mu, v)).simplify_full())
