# Figure: left, the Matsubara frequencies on the imaginary frequency axis, bosonic omega_n = 2 pi n/beta (filled) and
# fermionic omega_n = (2n + 1) pi/beta (diamonds), in units of pi/beta; right, a bosonic mode cos(omega_1 tau), periodic
# with period beta, and a fermionic mode cos(omega_0 tau), antiperiodic, on two turns of the thermal circle.
# Checks the periodicity, the orthogonality of the modes, the bosonic and fermionic frequency sums (by residues, with a
# numerical cross-check), the oscillator free energy from the bosonic sum, and the thermal Green's function.
# Output: figures/matsubara_frequencies.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "matsubara_frequencies.svg")
n, k, beta, w, z, a, tau = var("n k beta omega z a tau")
assume(beta > 0, w > 0, a > 0)

left = line([(0, -9.5), (0, 9.5)], color="gray") + line([(-1.5, 0), (1.5, 0)], color="gray")
left += point([(-0.4, 2 * j) for j in range(-4, 5)], color="royalblue", size=45, legend_label=r"bosons $2\pi n/\beta$")
left += point([(0.4, 2 * j + 1) for j in range(-5, 5)], color="firebrick", size=45, marker="D", legend_label=r"fermions $(2n+1)\pi/\beta$")
for j in range(-4, 5):
    left += text(r"$%d$" % (2 * j), (-1.0, 2 * j), color="royalblue", fontsize=9)
for j in range(-5, 5):
    left += text(r"$%d$" % (2 * j + 1), (1.0, 2 * j + 1), color="firebrick", fontsize=9)
left += text(r"$\mathrm{Im}\,\omega$  (units of $\pi/\beta$)", (0, 10.3), color="black", fontsize=10)
left.axes(False); left.set_legend_options(loc="lower right"); left.set_aspect_ratio("automatic")
left.set_axes_range(xmin=-1.6, xmax=1.6, ymin=-10, ymax=11)
B = 1
right = plot(cos(2 * pi * tau / B), (tau, 0, 2 * B), color="royalblue", thickness=2, legend_label=r"boson $\cos(2\pi\tau/\beta)$: $\phi(\tau + \beta) = \phi(\tau)$")
right += plot(cos(pi * tau / B), (tau, 0, 2 * B), color="firebrick", thickness=2, linestyle="--", legend_label=r"fermion $\cos(\pi\tau/\beta)$: $\psi(\tau + \beta) = -\psi(\tau)$")
right += line([(B, -1.2), (B, 1.2)], color="gray", linestyle=":")
right.axes_labels([r"$\tau$", ""]); right.set_legend_options(loc="lower left"); right.set_axes_range(ymin=-1.9, ymax=1.3)
graphics_array([left, right], 1, 2).save(OUT, figsize=[11, 4])

wB = 2 * pi * n / beta; wF = (2 * n + 1) * pi / beta
ok1 = all(bool(exp(-I * wB * beta).subs(n=j) == 1) and bool(exp(-I * wF * beta).subs(n=j) == -1) for j in range(-6, 7))
ok1 = ok1 and bool(exp(-I * wB * beta).simplify_full().subs(n=k) == exp(-2 * I * pi * k))
print("[PASS] periodicity: e^{-i omega_n beta} = 1 for omega_n = 2 pi n/beta and -1 for omega_n = (2n + 1) pi/beta (n = -6..6):", ok1)
orth = all(bool(integrate(exp(I * (2 * pi * (p - q) / beta) * tau), tau, 0, beta) / beta == (1 if p == q else 0)) for p in range(-2, 3) for q in range(-2, 3))
orthF = all(bool(integrate(exp(I * (2 * (p - q) * pi / beta) * tau), tau, 0, beta) / beta == (1 if p == q else 0)) for p in range(-2, 3) for q in range(-2, 3))
print("[PASS] orthonormality: (1/beta) int_0^beta e^{i (omega_p - omega_q) tau} d tau = delta_pq for the bosonic and the fermionic modes:", orth and orthF)

# frequency sums by residues: sum_n f(n) = -sum Res[pi cot(pi z) f(z)] and sum_n f(n + 1/2) = sum Res[pi tan(pi z) f(z)]
f = 1 / (z ** 2 + a ** 2)
sB = -sum((pi * cot(pi * z) * f).residue(z == s_) for s_ in [I * a, -I * a])
sF = sum((pi * tan(pi * z) * f).residue(z == s_) for s_ in [I * a, -I * a])
okr = bool((sB - pi * coth(pi * a) / a).simplify_full() == 0) and bool((sF - pi * tanh(pi * a) / a).simplify_full() == 0)
num = all(abs(sum(1.0 / (j ** 2 + av ** 2) for j in range(-20000, 20001)) + 2.0 / 20000 - (pi * coth(pi * av) / av).n()) < 1e-7 for av in [0.3, 1.0, 2.5])
numF = all(abs(sum(1.0 / ((j + 0.5) ** 2 + av ** 2) for j in range(-20000, 20000)) + 2.0 / 20000 - (pi * tanh(pi * av) / av).n()) < 1e-7 for av in [0.3, 1.0, 2.5])
print("[PASS] residue sums: sum_n 1/(n^2 + a^2) = pi coth(pi a)/a and sum_n 1/((n + 1/2)^2 + a^2) = pi tanh(pi a)/a (numerically checked with 40000 terms):", okr and num and numF)
# rescale: (1/beta) sum 1/(omega_n^2 + omega^2) with omega_n = 2 pi n/beta, i.e. a = beta omega/(2 pi)
TB = (beta / (4 * pi ** 2)) * (pi * coth(pi * a) / a).subs(a=beta * w / (2 * pi))
TF = (beta / (4 * pi ** 2)) * (pi * tanh(pi * a) / a).subs(a=beta * w / (2 * pi))
okB = bool((TB - coth(beta * w / 2) / (2 * w)).simplify_full() == 0) and bool((TF - tanh(beta * w / 2) / (2 * w)).simplify_full() == 0)
print("[PASS] Matsubara sums: (1/beta) sum_n 1/(omega_n^2 + omega^2) = coth(beta omega/2)/(2 omega) (bosons) and tanh(beta omega/2)/(2 omega) (fermions):", okB)
# oscillator free energy: d/d omega log Z = -sum_n omega/(omega_n^2 + omega^2) = -(beta/2) coth(beta omega/2) = d/d omega [-log(2 sinh(beta omega/2))]
lhs = -w * beta * coth(beta * w / 2) / (2 * w)
rhs = diff(-log(2 * sinh(beta * w / 2)), w)
print("[PASS] the oscillator from the thermal circle: d log Z/d omega = -omega sum_n 1/(omega_n^2 + omega^2) = d/d omega [-log(2 sinh(beta omega/2))], and the T -> 0 limit is the zero-point energy omega/2:", bool((lhs - rhs).simplify_full() == 0) and bool(limit(-log(2 * sinh(beta * w / 2)) / beta, beta=oo) == -w / 2))
# thermal Green's function on the circle
G = cosh(w * (tau - beta / 2)) / (2 * w * sinh(beta * w / 2))
okG = bool((-diff(G, tau, 2) + w ** 2 * G).simplify_full() == 0) and bool((G.subs(tau=0) - G.subs(tau=beta)).simplify_full() == 0) \
      and bool((diff(G, tau).subs(tau=0) - diff(G, tau).subs(tau=beta) + 1).simplify_full() == 0) and bool((G.subs(tau=0) - coth(beta * w / 2) / (2 * w)).simplify_full() == 0)
print("[PASS] thermal Green's function G(tau) = cosh(omega(tau - beta/2))/(2 omega sinh(beta omega/2)): (-d_tau^2 + omega^2) G = 0 on (0, beta), G(0) = G(beta) = the Matsubara sum, unit jump of G' at tau = 0:", okG)
# the analogy with a circle: eigenvalues of -d_tau^2 on the modes; tower of three-dimensional masses
mmass, Tt = var("m T"); assume(Tt > 0)
eig = bool((-diff(exp(-I * wB * tau), tau, 2) - wB ** 2 * exp(-I * wB * tau)).simplify_full() == 0) and bool((-diff(exp(-I * wF * tau), tau, 2) - wF ** 2 * exp(-I * wF * tau)).simplify_full() == 0)
M2 = mmass ** 2 + wB.subs(beta=1 / Tt) ** 2
print("[PASS] spectrum on the thermal circle: -d_tau^2 e^{-i omega_n tau} = omega_n^2 e^{-i omega_n tau}; masses M_n^2 = m^2 + (2 pi n T)^2 for bosons, the lowest fermionic value being (pi T)^2:", eig and bool(M2 == mmass ** 2 + 4 * pi ** 2 * n ** 2 * Tt ** 2) and bool(min((wF.subs(beta=1 / Tt) ** 2).subs(n=j) for j in range(-3, 3)) == pi ** 2 * Tt ** 2))
print("wrote", os.path.relpath(OUT))
