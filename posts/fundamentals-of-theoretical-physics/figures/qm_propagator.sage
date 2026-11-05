# Figure: the free propagator and the classical limit (units m = 1). Left: real and imaginary parts of
# K(x, 0; t) = sqrt(m/(2 pi i hbar t)) exp(i m x^2/(2 hbar t)) at t = 1, hbar = 1: a Gaussian kernel with imaginary
# width, oscillating faster and faster away from the classical path x = 0. Right: the phase factor
# Re exp(i (S[x_cl + a sin(pi s/T)] - S_cl)/hbar) along a one-parameter family of paths (T = 1) for hbar = 1, 0.2,
# 0.05: away from the classical path a = 0, where S is stationary, the phase oscillates and contributions cancel.
# Output: figures/qm_propagator.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qm_propagator.svg")
x, y, z = var("x y z", domain="real")
hbar, m, t, t1, t2, w, T = var("hbar m t t1 t2 w T", domain="positive")
K = lambda X, Y, tt: sqrt(m / (2 * pi * I * hbar * tt)) * exp(I * m * (X - Y) ** 2 / (2 * hbar * tt))
k0 = K(x, y, t)
print("[PASS] the free propagator K = sqrt(m/(2 pi i hbar t)) exp(i m (x - y)^2/(2 hbar t)) solves i hbar K_t = -(hbar^2/2m) K_xx:",
      bool(((I * hbar * diff(k0, t) + hbar ** 2 / (2 * m) * diff(k0, x, 2)) / k0).simplify_full() == 0))
tau, tau1, tau2 = var("tau tau1 tau2", domain="positive")
KE = lambda X, Y, tt: sqrt(m / (2 * pi * hbar * tt)) * exp(-m * (X - Y) ** 2 / (2 * hbar * tt))
comp = integrate(KE(x, z, tau1) * KE(z, y, tau2), z, -oo, oo)
print("[PASS] Euclidean composition: int K_E(x, z; tau1) K_E(z, y; tau2) dz = K_E(x, y; tau1 + tau2) (a Gaussian integral):",
      bool((comp / KE(x, y, tau1 + tau2)).canonicalize_radical() == 1))
A = -I * m / (2 * hbar) * (1 / t1 + 1 / t2); B = I * m / hbar * (x / t1 + y / t2); C = I * m / (2 * hbar) * (x ** 2 / t1 + y ** 2 / t2)
expo_ok = bool((B ** 2 / (4 * A) + C - I * m * (x - y) ** 2 / (2 * hbar * (t1 + t2))).simplify_full() == 0)
pref_ok = bool(((m / (2 * pi * I * hbar * t1)) * (m / (2 * pi * I * hbar * t2)) * (pi / A) - m / (2 * pi * I * hbar * (t1 + t2))).simplify_full() == 0)
print("[PASS] real-time composition by the Fresnel formula int exp(-A z^2 + B z + C) dz = sqrt(pi/A) exp(B^2/(4A) + C): exponent and squared prefactor of K(t1) * K(t2) match K(t1 + t2):", expo_ok and pref_ok)
ok = True
for NS in [2, 3, 4]:
    eps_ = tau / NS
    acc = KE(x, z, eps_)
    for j in range(NS - 1):
        zz = var("zz%d" % j)
        acc = integrate(acc.subs(z=zz) * KE(zz, z, eps_), zz, -oo, oo)
    ok = ok and bool((acc.subs(z=y) / KE(x, y, tau)).canonicalize_radical() == 1)
print("[PASS] the time-sliced Euclidean path integral with N = 2, 3, 4 slices, (m/(2 pi hbar eps))^(N/2) int prod dz_k exp(-sum m (z_(k+1) - z_k)^2/(2 hbar eps)), equals K_E(x, y; tau):", ok)
num = {hbar: 1, m: 1}
def evolve(xv, tv):
    re = numerical_integral(lambda u: (CDF(K(x, y, t).subs(num).subs(x=xv, y=u, t=tv)) * exp(-u ** 2 / 4) / (2 * pi) ** (1 / 4)).real(), -30, 30)[0]
    im = numerical_integral(lambda u: (CDF(K(x, y, t).subs(num).subs(x=xv, y=u, t=tv)) * exp(-u ** 2 / 4) / (2 * pi) ** (1 / 4)).imag(), -30, 30)[0]
    return CDF(re, im)
packet = lambda xv, tv: (2 * pi) ** (-1 / 4) * (1 + I * tv / 2) ** (-1 / 2) * exp(-xv ** 2 / (4 * (1 + I * tv / 2)))
print("[PASS] int K(x, y; t) psi_0(y) dy reproduces the spreading Gaussian packet (hbar = m = sigma = 1) at (x, t) = (0, 1), (1, 1), (2, 3) to 1e-8:",
      all(abs(evolve(xv, tv) - CDF(packet(xv, tv))) < 1e-8 for xv, tv in [(0, 1), (1, 1), (2, 3)]))
s, aa = var("s aa", domain="real")
xcl = (y * sin(w * (T - s)) + x * sin(w * s)) / sin(w * T)            # from y at s = 0 to x at s = T
Lag = lambda q: m / 2 * diff(q, s) ** 2 - m * w ** 2 / 2 * q ** 2
Scl = integrate(Lag(xcl), s, 0, T).simplify_full()
print("[PASS] the oscillator's classical path from y to x in time T has action S_cl = m omega ((x^2 + y^2) cos(omega T) - 2 x y)/(2 sin(omega T)):",
      bool((diff(xcl, s, 2) + w ** 2 * xcl).simplify_full() == 0) and bool((Scl - m * w * ((x ** 2 + y ** 2) * cos(w * T) - 2 * x * y) / (2 * sin(w * T))).simplify_full() == 0))
dS = (integrate(Lag(xcl + aa * sin(pi * s / T)), s, 0, T) - Scl).simplify_full()
print("[PASS] along x_cl + a sin(pi s/T), S - S_cl = m a^2 (pi^2 - omega^2 T^2)/(4T): no linear term in a, the classical path is stationary:",
      bool((dS - m * aa ** 2 * (pi ** 2 - w ** 2 * T ** 2) / (4 * T)).simplify_full() == 0))
Km = sqrt(m * w / (2 * pi * I * hbar * sin(w * t))) * exp(I * m * w * ((x ** 2 + y ** 2) * cos(w * t) - 2 * x * y) / (2 * hbar * sin(w * t)))
print("[PASS] the oscillator (Mehler) kernel sqrt(m omega/(2 pi i hbar sin omega t)) exp(i S_cl/hbar) solves i hbar K_t = -(hbar^2/2m) K_xx + (m omega^2 x^2/2) K:",
      bool(((I * hbar * diff(Km, t) + hbar ** 2 / (2 * m) * diff(Km, x, 2) - m * w ** 2 * x ** 2 / 2 * Km) / Km).simplify_full() == 0))
# the figure
u = var("u")
g1 = plot(lambda q: CDF(sqrt(1 / (2 * pi * I)) * exp(I * q ** 2 / 2)).real(), (-4.5, 4.5), color="royalblue", thickness=1.4, plot_points=800, legend_label=r"$\mathrm{Re}\,K$")
g1 += plot(lambda q: CDF(sqrt(1 / (2 * pi * I)) * exp(I * q ** 2 / 2)).imag(), (-4.5, 4.5), color="firebrick", thickness=1.4, plot_points=800, legend_label=r"$\mathrm{Im}\,K$")
g1.axes_labels([r"$x$", r"$K(x, 0; 1)$"]); g1.set_legend_options(font_size=8, loc="lower left")
g2 = Graphics()
for hb, c in zip(["1", "0.2", "0.05"], ["royalblue", "darkgreen", "firebrick"]):
    g2 += plot(cos(pi ** 2 * u ** 2 / (4 * RR(hb))), (u, -1, 1), color=c, thickness=1.2, plot_points=1500, legend_label=r"$\hbar = %s$" % hb)
g2.axes_labels([r"$a$", r"$\cos((S - S_{\rm cl})/\hbar)$"]); g2.set_legend_options(font_size=8, loc="lower left"); g2.set_axes_range(ymin=-1.6, ymax=1.1)
G = graphics_array([g1, g2])
G.matplotlib(figsize=[10, 3.8]).savefig(OUT, bbox_inches="tight")
print("wrote", os.path.relpath(OUT))
