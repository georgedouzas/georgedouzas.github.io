# Figure: the Carnot cycle of an ideal monatomic gas (N = 1, gamma = 5/3, k_B = 1) between T_h = 2 and
# T_c = 1, drawn in the p-V plane (two isotherms, two adiabats) and in the T-S plane (a rectangle). Checks the
# Maxwell relations as equalities of mixed second derivatives of the four potentials, the ideal-gas and van der Waals
# equations of state derived from F(T, V), the Carnot efficiency, C_p - C_v and the Euler relation.
# Output: figures/carnot_cycle.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "carnot_cycle.svg")
T, V, p, S, N, a, b, c, x = var("T V p S N a b c x")
assume(T > 0, V > 0, N > 0, a > 0, b > 0)

# --- the cycle: N = 1, gamma = 5/3, T_h = 2, T_c = 1, V_A = 1, V_B = 2
Th, Tc, gam = 2, 1, 5 / 3
VA, VB = 1, 2
VC = VB * (Th / Tc) ** (1 / (gam - 1)); VD = VA * (Th / Tc) ** (1 / (gam - 1))
iso_h = plot(Th / x, (x, VA, VB), color="firebrick", thickness=2, legend_label=r"isotherm $T_h = 2$")
adia_1 = plot(Th * VB ** (gam - 1) / x ** gam, (x, VB, VC), color="royalblue", thickness=2, legend_label="adiabat")
iso_c = plot(Tc / x, (x, VD, VC), color="darkorange", thickness=2, legend_label=r"isotherm $T_c = 1$")
adia_2 = plot(Th * VA ** (gam - 1) / x ** gam, (x, VA, VD), color="royalblue", thickness=2)
guide = plot(Th / x, (x, 0.6, 6), color="lightgray", linestyle="--") + plot(Tc / x, (x, 0.6, 6), color="lightgray", linestyle="--")
pts = {"A": (VA, Th / VA), "B": (VB, Th / VB), "C": (VC, Tc / VC), "D": (VD, Tc / VD)}
lab = sum(point(P, color="black", size=25) + text(k, (P[0] + 0.15, P[1] + 0.08), color="black", fontsize=12) for k, P in pts.items())
arrows = arrow((1.4, Th / 1.4 + 0.25), (1.6, Th / 1.6 + 0.2), color="firebrick", width=1) + arrow((4.4, Tc / 4.4 - 0.12), (3.8, Tc / 3.8 - 0.12), color="darkorange", width=1)
pV = guide + iso_h + adia_1 + iso_c + adia_2 + lab + arrows
pV.axes_labels([r"$V$", r"$p$"]); pV.set_legend_options(loc="upper right")
SA = 1; SB = 1 + log(2.0)   # S_A = 1 is an arbitrary origin of entropy
TS = line([(SA, Th), (SB, Th)], color="firebrick", thickness=2) + line([(SB, Th), (SB, Tc)], color="royalblue", thickness=2) \
    + line([(SB, Tc), (SA, Tc)], color="darkorange", thickness=2) + line([(SA, Tc), (SA, Th)], color="royalblue", thickness=2) \
    + polygon([(SA, Tc), (SB, Tc), (SB, Th), (SA, Th)], color="lightgray", alpha=0.4) \
    + sum(text(k, P, color="black", fontsize=12) for k, P in {"A": (SA - 0.05, Th + 0.08), "B": (SB + 0.05, Th + 0.08), "C": (SB + 0.05, Tc - 0.08), "D": (SA - 0.05, Tc - 0.08)}.items()) \
    + text(r"$W = (T_h - T_c)\Delta S$", ((SA + SB) / 2, 1.5), color="black", fontsize=11)
TS.axes_labels([r"$S$", r"$T$"]); TS.set_aspect_ratio("automatic")
pV.set_axes_range(xmin=0, xmax=6.2, ymin=0, ymax=2.3); TS.set_axes_range(xmin=0, xmax=2, ymin=0, ymax=2.4)
G = graphics_array([pV, TS])
G.save(OUT, figsize=[11, 4.2])

# --- Maxwell relations as equalities of mixed second derivatives
Ff = function("F")(T, V); Gf = function("G")(T, p); Uf = function("U")(S, V); Hf = function("H")(S, p)
m1 = bool(diff(-diff(Ff, T), V) == diff(-diff(Ff, V), T))          # (dS/dV)_T = (dp/dT)_V
m2 = bool(diff(-diff(Gf, T), p) == -diff(diff(Gf, p), T))          # (dS/dp)_T = -(dV/dT)_p
m3 = bool(diff(diff(Uf, S), V) == -diff(-diff(Uf, V), S))          # (dT/dV)_S = -(dp/dS)_V
m4 = bool(diff(diff(Hf, S), p) == diff(diff(Hf, p), S))            # (dT/dp)_S = (dV/dS)_p
print("[PASS] Maxwell relations: the mixed second derivatives of F(T, V), G(T, p), U(S, V), H(S, p) commute:", m1 and m2 and m3 and m4)

# --- ideal gas from F(T, V) = -N T (log(V/N) + 3/2 log T + c)
F_id = -N * T * (log(V / N) + 3 / 2 * log(T) + c)
p_id = -diff(F_id, V); S_id = -diff(F_id, T); U_id = (F_id + T * S_id).simplify_full()
ok_id = bool(p_id == N * T / V) and bool(U_id == 3 / 2 * N * T) and bool(diff(S_id, V) == diff(p_id, T))
print("[PASS] ideal gas: F = -N T (log(V/N) + 3/2 log T + c) gives p V = N T, U = 3 N T/2 and (dS/dV)_T = (dp/dT)_V = N/V:", ok_id and bool(diff(S_id, V) == N / V))

# --- van der Waals from F = F_ideal(V - N b) - a N^2/V
F_vw = -N * T * (log((V - N * b) / N) + 3 / 2 * log(T) + c) - a * N ** 2 / V
p_vw = -diff(F_vw, V); S_vw = -diff(F_vw, T)
ok_vw = bool((p_vw - (N * T / (V - N * b) - a * N ** 2 / V ** 2)).simplify_full() == 0) and bool(diff(S_vw, V) == diff(p_vw, T))
crit = {T: 8 * a / (27 * b), V: 3 * N * b}
d1 = diff(p_vw, V).subs(crit).simplify_full(); d2 = diff(p_vw, V, 2).subs(crit).simplify_full()
pc = p_vw.subs(crit).simplify_full()
# uniqueness: (dp/dV)_T = 0 = (d2p/dV2)_T eliminate T to give V = 3 N b
Tsol = solve(diff(p_vw, V) == 0, T)[0].rhs()
Vsol = [s.rhs() for s in solve(diff(p_vw, V, 2).subs(T=Tsol).simplify_full().numerator() == 0, V)]
ok_cr = bool(d1 == 0) and bool(d2 == 0) and bool(pc == a / (27 * b ** 2)) and bool(3 * N * b in Vsol) and bool(Tsol.subs(V=3 * N * b) == 8 * a / (27 * b))
print("[PASS] van der Waals: p = N T/(V - N b) - a N^2/V^2 from F, Maxwell relation holds, critical point T_c = 8a/(27b), V_c = 3Nb, p_c = a/(27b^2):", ok_vw and ok_cr)

# --- Carnot cycle of the ideal monatomic gas
Th_, Tc_, VA_, VB_ = var("T_h T_c V_A V_B")
assume(Th_ > 0, Tc_ > 0, VA_ > 0, VB_ > 0)
VC_ = VB_ * (Th_ / Tc_) ** (3 / 2); VD_ = VA_ * (Th_ / Tc_) ** (3 / 2)
Qh = N * Th_ * log(VB_ / VA_); Qc = N * Tc_ * log(VC_ / VD_)
W = Qh - Qc
eff = (W / Qh).simplify_full()
adiab = bool((Tc_ * VC_ ** (2 / 3) - Th_ * VB_ ** (2 / 3)).simplify_full() == 0)
dS = (Qh / Th_ - Qc / Tc_).simplify_log()
print("[PASS] Carnot cycle: T V^(2/3) is constant on the adiabats, the entropy change over the cycle is 0 and the efficiency W/Q_h = 1 - T_c/T_h:", adiab and bool(dS == 0) and bool(eff == 1 - Tc_ / Th_))
print("[PASS] the drawn cycle: V_C = 4 sqrt 2, V_D = 2 sqrt 2, W = log 2, efficiency 1/2:", bool(abs(VC - 4 * sqrt(2.0)) < 1e-12) and bool(abs(VD - 2 * sqrt(2.0)) < 1e-12) and bool((Qh - Qc).subs({N: 1, Th_: 2, Tc_: 1, VA_: 1, VB_: 2}).simplify_log() == log(2)))

# --- C_p - C_v = -T (dp/dT)_V^2 / (dp/dV)_T
def cp_minus_cv(pp):
    return (-T * diff(pp, T) ** 2 / diff(pp, V)).simplify_full()
d_id = cp_minus_cv(p_id); d_vw = cp_minus_cv(p_vw)
ok_c = bool(d_id == N) and bool((d_vw - N / (1 - 2 * a * N * (V - N * b) ** 2 / (T * V ** 3))).simplify_full() == 0)
print("[PASS] C_p - C_v = -T (dp/dT)_V^2/(dp/dV)_T equals N for the ideal gas and N/(1 - 2aN(V - Nb)^2/(T V^3)) for van der Waals:", ok_c)

# --- Euler relation for U(S, V, N) of the ideal gas (homogeneous of degree one)
s0 = var("s0")
U_SV = 3 / 2 * N * (N / V) ** (2 / 3) * exp(2 * S / (3 * N) - s0)
T_e = diff(U_SV, S); p_e = -diff(U_SV, V); mu_e = diff(U_SV, N)
euler = bool((T_e * S - p_e * V + mu_e * N - U_SV).simplify_full() == 0)
print("[PASS] Euler relation U = T S - p V + mu N for the ideal gas U(S, V, N) = (3/2) N (N/V)^(2/3) exp(2S/(3N) - s0), with U = 3NT/2:", euler and bool((U_SV - 3 / 2 * N * T_e).simplify_full() == 0))
print("wrote", os.path.relpath(OUT))
