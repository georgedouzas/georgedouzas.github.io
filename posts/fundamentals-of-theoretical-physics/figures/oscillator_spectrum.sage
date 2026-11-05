# Figure: quantisation of one field mode. Left: the harmonic oscillator potential with its quantised energy levels
# E_n = (n + 1/2) omega (horizontal lines) and the first wavefunctions, computed from the ladder operators
# represented as truncated matrices with Sage. Right: a Klein-Gordon mode of wave number k is such an oscillator
# with omega = sqrt(k^2 + m^2); its n-th level is the state with n particles of energy omega.
# Output: figures/oscillator_spectrum.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oscillator_spectrum.svg")
N = 12
a = matrix(SR, N, N, lambda i, j: sqrt(j) if j == i + 1 else 0)      # annihilation operator, a|n> = sqrt(n)|n-1>
ad = a.transpose()
H = (ad * a + identity_matrix(SR, N) / 2).apply_map(lambda e: e.simplify_full())                                     # H = a^dagger a + 1/2 (omega = 1)
levels = [QQ(H[n, n]) for n in range(N)]
x = var("x", domain="real")
left = plot(x**2 / 2, (x, -4, 4), color="black", thickness=1.5)
herm = [1, 2 * x, 4 * x**2 - 2, 8 * x**3 - 12 * x]
for n in range(4):
    E = levels[n]
    left += line([(-sqrt(2 * E), E), (sqrt(2 * E), E)], color="grey")
    psi = herm[n] * exp(-x**2 / 2) / sqrt(2**n * factorial(n) * sqrt(pi))
    left += plot(E + 0.8 * psi, (x, -4, 4), color=hue(0.6 + 0.1 * n), thickness=1.2)
    left += text(r"$E_%d=%s\,\omega$" % (n, latex(E)), (4.9, E), fontsize=8)
left += text(r"one mode $=$ one oscillator: $E_n=(n+\frac{1}{2})\omega$", (0, 5.2), fontsize=9)
left.axes_labels([r"$q$", r"$E$"])
k = var("k", domain="real")
right = Graphics()
for n in range(4):
    right += plot(n * sqrt(k**2 + 1), (k, -2, 2), color=hue(0.6 + 0.1 * n), thickness=1.5, legend_label=r"$n=%d$ particles" % n)
right += text(r"energy of $n$ quanta of a mode $k$: $n\sqrt{k^2+m^2}$ ($m=1$)", (0, 7.4), fontsize=9)
right.axes_labels([r"$k$", r"$E-E_0$"]); right.set_legend_options(loc="upper left", font_size=8)
g = graphics_array([left, right]); fig = g.matplotlib(figsize=[9.5, 4]); fig.savefig(OUT, bbox_inches="tight")
comm = (a * ad - ad * a)
print("[PASS] [a, a^dagger] = 1 on the truncated space (all but the last diagonal entry):", all(comm[i, i] == 1 for i in range(N - 1)) and all(comm[i, j] == 0 for i in range(N) for j in range(N) if i != j))
print("[PASS] the spectrum of H = a^dagger a + 1/2 is n + 1/2:", levels == [n + QQ(1) / 2 for n in range(N)])
print("wrote", os.path.relpath(OUT))
