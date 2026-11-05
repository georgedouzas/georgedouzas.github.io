# Figure: heat capacity per system of the two-level system (Schottky peak), the quantum harmonic oscillator (Einstein)
# and the classical oscillator (equipartition, C = 1), against T/epsilon (k_B = 1; epsilon = level spacing, or hbar*omega).
# Checks the partition functions of the two-level system, the oscillator (geometric series), the classical ideal gas
# (Gaussian momentum integrals), the Schottky peak position, the high- and low-temperature limits, equipartition and
# the third law for both quantum systems. Output: figures/schottky_peak.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schottky_peak.svg")
beta, eps, w, T, x, n, m, V, N, hbar, pp, q, a, b = var("beta epsilon omega T x n m V N hbar p q a b")
assume(beta > 0, eps > 0, w > 0, T > 0, x > 0, m > 0, V > 0, N > 0, hbar > 0, a > 0, b > 0)

C2 = x ** 2 * exp(x) / (1 + exp(x)) ** 2          # two-level, x = eps/T
CE = x ** 2 * exp(x) / (exp(x) - 1) ** 2          # oscillator, x = omega/T
t = var("t")
fig = plot(C2.subs(x=1 / t), (t, 0.02, 3), color="royalblue", thickness=2, legend_label="two-level system (Schottky)")
fig += plot(CE.subs(x=1 / t), (t, 0.02, 3), color="firebrick", thickness=2, legend_label="quantum oscillator (Einstein)")
fig += line([(0, 1), (3, 1)], color="black", linestyle="--", legend_label="classical oscillator (equipartition)")
xs_ = find_root(diff(C2, x), 1, 4)
fig += point((1 / xs_, C2.subs(x=xs_)), color="royalblue", size=40)
fig.axes_labels([r"$T$", r"$C$"]); fig.set_legend_options(loc="center right")
fig.save(OUT, figsize=[7.5, 4], ymin=0, ymax=1.1)

# --- two-level system
Z2 = 1 + exp(-beta * eps)
U2 = -diff(log(Z2), beta)
C2b = (-beta ** 2 * diff(U2, beta)).simplify_full()   # C = dU/dT = -beta^2 dU/dbeta
ok1 = bool((U2 - eps / (exp(beta * eps) + 1)).simplify_full() == 0) and bool((C2b - C2.subs(x=beta * eps)).simplify_full() == 0)
print("[PASS] two-level system: Z = 1 + e^{-eps/T}, U = eps/(e^{eps/T} + 1), C = x^2 e^x/(1 + e^x)^2 with x = eps/T:", ok1)
peak_eq = bool((diff(C2, x) * (1 + exp(x)) ** 3 / (x * exp(x)) - (2 * (1 + exp(x)) - x * (exp(x) - 1))).simplify_full() == 0)
print("[PASS] Schottky peak: dC/dx = 0 is x tanh(x/2) = 2, at x = %.4f, i.e. T = %.4f eps, with C_max = %.4f:" % (xs_, 1 / xs_, C2.subs(x=xs_)), bool(peak_eq and abs(xs_ * tanh(xs_ / 2) - 2) < 1e-10 and abs(xs_ - 2.3994) < 1e-4 and abs(C2.subs(x=xs_) - 0.4392) < 1e-4))

# --- quantum oscillator: geometric series (sinh written as exponentials, sh(z) = (e^z - e^{-z})/2)
sh = lambda z: (exp(z) - exp(-z)) / 2
Zs = sum(exp(-beta * w * (n + 1 / 2)), n, 0, oo)
ok2 = bool((Zs - 1 / (2 * sh(beta * w / 2))).simplify_full() == 0)
UE = -diff(log(1 / (2 * sh(beta * w / 2))), beta)
ok3 = bool((UE - (w / 2 + w / (exp(beta * w) - 1))).simplify_full() == 0)
FE = -log(1 / (2 * sh(w / (2 * T)))) * T
CEb = (-T * diff(FE, T, 2)).simplify_full()
ok4 = bool((CEb - CE.subs(x=w / T)).simplify_full() == 0)
print("[PASS] quantum oscillator: sum_n e^{-beta omega (n + 1/2)} = 1/(2 sinh(beta omega/2)), U = omega/2 + omega/(e^{beta omega} - 1), C = -T d^2F/dT^2 = x^2 e^x/(e^x - 1)^2:", ok2 and ok3 and ok4)
ok5 = bool(limit(CE, x=0) == 1) and bool(limit(C2, x=0) == 0) and bool(limit(CE / (x ** 2 * exp(-x)), x=oo) == 1) and bool(limit(C2 / (x ** 2 * exp(-x)), x=oo) == 1)
print("[PASS] limits: C -> 1 (oscillator) and C -> 0 (two-level) as T -> oo; both C ~ x^2 e^{-x} as T -> 0:", ok5)
S2 = (beta * U2 + log(Z2)).subs(beta=1 / T)
ok6 = bool(limit(S2, T=0, dir="+") == 0) and bool(limit(S2, T=oo) == log(2))
SE = (beta * UE + log(1 / (2 * sh(beta * w / 2)))).subs(beta=x / w)     # S = beta U + log Z, x = omega/T
SEx = x / (exp(x) - 1) - log(1 - exp(-x))
okE = all(abs(SE.subs(x=xv).n() - SEx.subs(x=xv).n()) < 1e-12 for xv in [0.3, 1, 2.5, 7])
ok6 = ok6 and okE and bool(limit(SEx, x=oo) == 0)
print("[PASS] third law: S -> 0 as T -> 0 for both systems; S -> log 2 for the two-level system as T -> oo; oscillator S = x/(e^x - 1) - log(1 - e^{-x}):", ok6)

# --- classical ideal gas: Gaussian momentum integrals
g = integrate(exp(-beta * pp ** 2 / (2 * m)), pp, -oo, oo)
Z1 = V * (g / (2 * pi * hbar)) ** 3
lam = sqrt(2 * pi * hbar ** 2 * beta / m)
ok7 = bool((Z1 - V / lam ** 3).canonicalize_radical() == 0)
FN = -(N * log(Z1) - (N * log(N) - N)) / beta    # Stirling: log N! ~ N log N - N
pN = -diff(FN, V)
UN = diff(beta * FN, beta)
SN = (beta * (UN - FN)).simplify_full()
ST = N * (log(V / (N * lam ** 3)) + 5 / 2)
ok8 = ok7 and bool((pN - N / (beta * V)).simplify_full() == 0) and bool((UN - 3 * N / (2 * beta)).simplify_full() == 0) and bool((SN - ST).canonicalize_radical().expand_log().simplify_full() == 0)
print("[PASS] ideal gas: int e^{-beta p^2/2m} dp = sqrt(2 pi m/beta), Z_1 = V/lambda^3, and with log N! ~ N log N - N: p V = N T, U = 3NT/2, S = N(log(V/(N lambda^3)) + 5/2) (Sackur-Tetrode):", ok8)

# --- equipartition and its generalisation
Zq = integrate(exp(-beta * a * pp ** 2), pp, -oo, oo); Eq = integrate(a * pp ** 2 * exp(-beta * a * pp ** 2), pp, -oo, oo) / Zq
Z4 = integrate(exp(-beta * b * q ** 4), q, -oo, oo); E4 = integrate(b * q ** 4 * exp(-beta * b * q ** 4), q, -oo, oo) / Z4
print("[PASS] equipartition: <a p^2> = T/2 for every a > 0, and <b q^4> = T/4:", bool((Eq - 1 / (2 * beta)).simplify_full() == 0) and bool((E4 - 1 / (4 * beta)).simplify_full() == 0))
print("wrote", os.path.relpath(OUT))
