# Figure: the particle in a box of width L. Left: the eigenfunctions phi_n = sqrt(2/L) sin(n pi x/L), n = 1..4,
# drawn at the height of their energies E_n = n^2 pi^2 hbar^2/(2 m L^2) inside the infinite well; right: the
# probability densities |phi_n|^2. Units hbar = m = L = 1 in the plot; the checks keep hbar, m, L symbolic.
# Output: figures/qm_particle_box.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qm_particle_box.svg")
x, t = var("x t", domain="real")
hbar, m, L = var("hbar m L", domain="positive")
phi = lambda n: sqrt(2 / L) * sin(n * pi * x / L)
En = lambda n: n ** 2 * pi ** 2 * hbar ** 2 / (2 * m * L ** 2)
print("[PASS] -hbar^2/(2m) phi_n'' = E_n phi_n with E_n = n^2 pi^2 hbar^2/(2 m L^2) and phi_n(0) = phi_n(L) = 0, n = 1..6:",
      all(bool((-hbar ** 2 / (2 * m) * diff(phi(n), x, 2) - En(n) * phi(n)).simplify_full() == 0) and phi(n).subs(x=0) == 0 and bool(phi(n).subs(x=L) == 0) for n in range(1, 7)))
gram = matrix(SR, 5, 5, lambda i, j: integrate(phi(i + 1) * phi(j + 1), x, 0, L).simplify_full())
print("[PASS] the box eigenfunctions phi_1..phi_5 are orthonormal in L^2(0, L):", gram == identity_matrix(5))
psi0 = sqrt(30 / L ** 5) * x * (L - x)
print("[PASS] psi = sqrt(30/L^5) x (L - x) is normalised:", bool(integrate(psi0 ** 2, x, 0, L).simplify_full() == 1))
cs = [integrate(phi(n) * psi0, x, 0, L) for n in range(1, 8)]
print("[PASS] its box coefficients are c_n = 8 sqrt(15)/(n^3 pi^3) for odd n and 0 for even n (n <= 7):",
      all(bool((cs[n - 1] - (8 * sqrt(15) / (n ** 3 * pi ** 3) if n % 2 else 0)).canonicalize_radical() == 0) for n in range(1, 8)))
k = var("k")
tot = sum(960 / ((2 * k + 1) ** 6 * pi ** 6), k, 0, oo)
print("[PASS] the probabilities |c_n|^2 = 960/(n^6 pi^6) over odd n sum to 1, and P(E_1) = 960/pi^6 = 0.99856:", bool(SR(tot).simplify_full() == 1) and bool(abs((960 / pi ** 6).n() - 0.99856) < 1e-5))
w = (En(2) - En(1)) / hbar
Psi = (phi(1) * exp(-I * En(1) * t / hbar) + phi(2) * exp(-I * En(2) * t / hbar)) / sqrt(2)
dens = (Psi * Psi.conjugate()).simplify_full()
mean = integrate(x * dens, x, 0, L).simplify_full()
print("[PASS] for (phi_1 + phi_2)/sqrt 2, <x>(t) = L/2 - (16 L/(9 pi^2)) cos(omega t) with omega = (E_2 - E_1)/hbar = 3 pi^2 hbar/(2 m L^2):",
      bool((mean - (L / 2 - 16 * L / (9 * pi ** 2) * cos(w * t))).exponentialize().simplify_full() == 0) and bool(w == 3 * pi ** 2 * hbar / (2 * m * L ** 2)))
print("[PASS] i hbar d/dt Psi = H Psi for that superposition:", bool((I * hbar * diff(Psi, t) + hbar ** 2 / (2 * m) * diff(Psi, x, 2)).simplify_full() == 0))
# the figure (hbar = m = L = 1, energies in units of E_1 scaled down for display)
xx = var("xx")
cols = ["firebrick", "royalblue", "darkgreen", "darkorange"]
g1 = line([(0, 24), (0, 0), (1, 0), (1, 24)], color="black", thickness=2.5)
g2 = Graphics()
for n, c in zip(range(1, 5), cols):
    e = n ** 2
    g1 += line([(0, e), (1, e)], color="gray", linestyle="--", thickness=0.8)
    g1 += plot(e + 0.9 * sqrt(2) * sin(n * pi * xx), (xx, 0, 1), color=c, thickness=1.8, legend_label=r"$\phi_%d$, $E_%d = %d E_1$" % (n, n, e))
    g2 += plot(2 * sin(n * pi * xx) ** 2, (xx, 0, 1), color=c, thickness=1.6, legend_label=r"$|\phi_%d|^2$" % n)
g1.axes_labels([r"$x/L$", r"$E/E_1$"]); g2.axes_labels([r"$x/L$", r"$L\,|\phi_n|^2$"])
g1.set_legend_options(loc="upper center", font_size=8, ncol=2); g2.set_legend_options(font_size=8)
G = graphics_array([g1, g2])
G.matplotlib(figsize=[10, 4]).savefig(OUT, bbox_inches="tight")
print("wrote", os.path.relpath(OUT))
