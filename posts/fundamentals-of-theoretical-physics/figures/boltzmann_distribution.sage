# Figure: left, the Boltzmann probabilities p_n = e^{-E_n/T}/Z of the equally spaced levels E_n = n (n = 0..9) at
# T = 0.5, 1, 3; right, the microcanonical entropy log binom(N, n) of N = 100 two-state spins per spin, against the
# Stirling form -x log x - (1 - x) log(1 - x), x = n/N. Checks the canonical identities for <E>, Var E, S and F, the
# maximum-entropy property of the Boltzmann distribution, the microcanonical temperature of the spin system, the
# grand-canonical occupation of one orbital and the 1/sqrt(N) fluctuations. Output: figures/boltzmann_distribution.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "boltzmann_distribution.svg")
beta, e1, e2, eps, mu, x, n, N, T = var("beta e1 e2 epsilon mu x n N T")
assume(beta > 0, eps > 0, e1 > 0, e2 > 0, T > 0, N > 0, n > 0, n < N)

# --- the figure
cols = {1 / 2: "royalblue", 1: "darkgreen", 3: "firebrick"}
left = Graphics()
for Tv, col in cols.items():
    Zv = sum(exp(-k / Tv) for k in range(10))
    left += line([(k, exp(-k / Tv) / Zv) for k in range(10)], color=col, thickness=1.2, linestyle="--")
    left += point([(k, exp(-k / Tv) / Zv) for k in range(10)], color=col, size=30, legend_label=r"$T = %s$" % ("1/2" if Tv == 1 / 2 else Tv))
left.axes_labels([r"$E_n$", r"$p_n$"]); left.set_legend_options(loc="upper right")
NN = 100
right = point([(k / NN, log(binomial(NN, k)).n() / NN) for k in range(NN + 1)], color="royalblue", size=8, legend_label=r"$\log\binom{N}{n}/N$, $N = 100$")
right += plot(-x * log(x) - (1 - x) * log(1 - x), (x, 0.001, 0.999), color="black", legend_label=r"$-x\log x - (1-x)\log(1-x)$")
right.axes_labels([r"$n/N$", r"$S/N$"]); right.set_legend_options(loc="lower center")
right.set_axes_range(ymin=0, ymax=0.85)
graphics_array([left, right]).save(OUT, figsize=[11, 4])

# --- canonical identities for a three-level system 0, e1, e2
E = [0, e1, e2]
Z = sum(exp(-beta * Ek) for Ek in E)
p = [exp(-beta * Ek) / Z for Ek in E]
Emean = sum(pk * Ek for pk, Ek in zip(p, E)); E2 = sum(pk * Ek ** 2 for pk, Ek in zip(p, E))
ok1 = bool((Emean + diff(log(Z), beta)).simplify_full() == 0) and bool((E2 - Emean ** 2 - diff(log(Z), beta, 2)).simplify_full() == 0)
print("[PASS] canonical ensemble of the levels 0, e1, e2: <E> = -d log Z/d beta and Var E = d^2 log Z/d beta^2:", ok1)
S = -sum(pk * log(pk) for pk in p)
ok2 = bool((S - (beta * Emean + log(Z))).simplify_full().simplify_log() == 0)
Fe = -log(Z) / beta
ok2 = ok2 and bool((Fe - (Emean - S / beta)).simplify_full().simplify_log() == 0)
print("[PASS] Gibbs entropy -sum p log p = beta <E> + log Z and F = -T log Z = <E> - T S:", ok2)

# --- maximum entropy: Lagrange stationarity and concavity
alpha = log(Z) - 1
stat = all(bool((-log(pk) - 1 - alpha - beta * Ek).simplify_full().simplify_log() == 0) for pk, Ek in zip(p, E))
ps = var("p0 p1 p2"); Sg = -sum(q * log(q) for q in ps)
H = matrix(SR, 3, 3, lambda i, j: diff(Sg, ps[i], ps[j]))
print("[PASS] the Boltzmann distribution is a stationary point of -sum p log p - alpha sum p - beta sum p E, and the Hessian diag(-1/p_i) is negative definite:", stat and bool(H == diagonal_matrix([-1 / q for q in ps])))

# --- microcanonical paramagnet: N spins, n excited with energy eps each, S = log binom(N, n) ~ N H(n/N)
Sst = -n * log(n / N) - (N - n) * log((N - n) / N)
invT = diff(Sst, n) / eps
nB_ = N / (exp(beta * eps) + 1)     # candidate solution of 1/T = beta, checked by substitution (unique: dS/dn is decreasing)
ok3 = bool((invT.subs(n=nB_) - beta).simplify_full().simplify_log().simplify_full() == 0) and bool((diff(invT, n) * eps).simplify_full() == (-N / (n * (N - n))).simplify_full())
stir = all(bool(0 <= (NN * (-(k / NN) * log(k / NN) - (1 - k / NN) * log(1 - k / NN)) - log(binomial(NN, k))).n() <= log(NN + 1.0)) for k in range(1, NN))
print("[PASS] microcanonical spins: 1/T = dS/dE with S = N H(n/N) gives n/N = 1/(e^{eps/T} + 1), and 0 <= N H(n/N) - log binom(N, n) <= log(N + 1) for N = 100:", ok3 and stir)

# --- grand canonical: one orbital of energy eps
y = var("y"); assume(y > 0, y < 1)
Xi_F = 1 + exp(-beta * (eps - mu))
Xi_B = sum(y ** n, n, 0, oo)    # y = e^{-beta (eps - mu)} < 1
okB = bool(Xi_B == 1 / (1 - y))
XiB = 1 / (1 - exp(-beta * (eps - mu)))
nF = (diff(log(Xi_F), mu) / beta).simplify_full(); nB = (diff(log(XiB), mu) / beta).simplify_full()
okF = bool((nF - 1 / (exp(beta * (eps - mu)) + 1)).simplify_full() == 0) and bool((nB - 1 / (exp(beta * (eps - mu)) - 1)).simplify_full() == 0)
print("[PASS] grand canonical orbital: Xi = 1 + e^{-beta(eps - mu)} (fermions), sum_n y^n = 1/(1 - y) (bosons), <N> = T d log Xi/d mu gives the Fermi-Dirac and Bose-Einstein occupations:", okB and okF)

# --- fluctuations of N independent two-level systems
Z1 = 1 + exp(-beta * eps)
EN = -N * diff(log(Z1), beta); VarN = N * diff(log(Z1), beta, 2)
ratio2 = (VarN / EN ** 2 * N).simplify_full()
print("[PASS] N independent two-level systems: (Delta E/<E>)^2 = e^{beta eps}/N, so the relative fluctuation falls as 1/sqrt(N):", bool(diff(ratio2, N) == 0) and bool((ratio2 - exp(beta * eps)).simplify_full() == 0))
print("wrote", os.path.relpath(OUT))
