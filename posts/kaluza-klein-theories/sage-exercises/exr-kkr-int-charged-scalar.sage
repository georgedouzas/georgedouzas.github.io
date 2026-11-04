# Exercise exr-kkr-int-charged-scalar: the tower of a five-dimensional massless complex scalar in a background gauge
# field. On g = eta (signature (+,-,-,-)), phi = 1 and constant A_mu, the wave equation Box_5 Phi = 0 on
# Phi = f(t, x) e^{i n y/R} is the four-dimensional equation (D_mu D^mu + (n/R)^2) f = 0 with D_mu = d_mu - i n A_mu/R;
# a gauge transformation y' = y + eps with eps = c_mu x^mu removes A, and f picks up the phase e^{-i n eps/R}.
M = Manifold(5, "M", structure="Lorentzian"); X = M.chart("t x v z y"); t, x, v, z, y = X[:]
a0, a1, n, R = var("a0 a1 n R"); assume(R > 0); eta = [1, -1, -1, -1]; A = [a0, a1, 0, 0]
G = M.metric("G")
for i in range(4):
    for j in range(i, 4): G[i, j] = (eta[i] if i == j else 0) - A[i] * A[j]
    G[i, 4] = -A[i]
G[4, 4] = -1
f = function("f")(t, x)
P = M.scalar_field(f * exp(I * n * y / R))
nab = G.connection()
red = (nab(nab(P).up(G)).trace().expr() / exp(I * n * y / R)).simplify_full()
D = lambda h, mu: diff(h, X[mu]) - I * n * A[mu] / R * h
DD = sum(eta[mu] * D(D(f, mu), mu) for mu in range(2))
print("[PASS] Box_5 Phi e^{-iny/R} = (D_mu D^mu + (n/R)^2) f with D = d - i n A/R:", bool((red - (DD + (n / R)^2 * f)).simplify_full() == 0))
fp = f * exp(-I * n * (a0 * t + a1 * x) / R)                # the gauge-transformed field, A' = A - d eps = 0
lhs = (diff(fp, t, 2) - diff(fp, x, 2) + (n / R)^2 * fp) / exp(-I * n * (a0 * t + a1 * x) / R)
print("[PASS] with eps = a_mu x^mu: (Box + (n/R)^2) (e^{-i n eps/R} f) = e^{-i n eps/R} (D D + (n/R)^2) f:", bool((lhs - (DD + (n / R)^2 * f)).simplify_full() == 0))
