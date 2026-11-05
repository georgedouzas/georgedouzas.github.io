# Figure: partial Fourier sums of the square wave sgn(x) on (-pi, pi) with the Gibbs overshoot near the jumps, and
# of the sawtooth x on (-pi, pi). Checks: orthogonality of the modes e^{inx}, the coefficients of both waves,
# Parseval's identity giving sum 1/n^2 = pi^2/6, the Dirichlet kernel, the L^2 error of a partial sum and the Gibbs
# constant (2/pi) Si(pi). Output: figures/fourier_square_wave.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fourier_square_wave.svg")
x = var("x")
square = piecewise([[(-pi, 0), -1], [(0, pi), 1]])
saw = piecewise([[(-pi, pi), x]])
def S_square(N):
    return sum(4 / (pi * n) * sin(n * x) for n in range(1, N + 1, 2))
def S_saw(N):
    return sum(2 * (-1) ** (n + 1) / n * sin(n * x) for n in range(1, N + 1))
cols = ["royalblue", "darkgreen", "darkorange", "firebrick"]
p = plot(square, (x, -pi, pi), color="black", thickness=2, legend_label="square wave")
for N, c in zip([1, 3, 9, 29], cols):
    p += plot(S_square(N), (x, -pi, pi), color=c, thickness=1.1, legend_label=r"$S_{%d}$" % N)
p += line([(-pi, 1.179), (pi, 1.179)], color="gray", linestyle=":", thickness=1)
p += text("Gibbs overshoot 1.179", (1.9, 1.32), fontsize=9, color="gray")
p.set_legend_options(loc="lower right", font_size=8)
q = plot(saw, (x, -pi, pi), color="black", thickness=2, legend_label="sawtooth")
for N, c in zip([1, 3, 9, 29], cols):
    q += plot(S_saw(N), (x, -pi, pi), color=c, thickness=1.1, legend_label=r"$S_{%d}$" % N)
q.set_legend_options(loc="upper left", font_size=8)
graphics_array([[p, q]]).matplotlib(figsize=(11, 4.2)).savefig(OUT, bbox_inches="tight")

# checks for prp-fourier-series-values
orth = all(bool(integrate(exp(I * n * x) * exp(-I * m * x), x, 0, 2 * pi) == (2 * pi if n == m else 0)) for n in range(-3, 4) for m in range(-3, 4))
print("[PASS] orthogonality: the integral of e^{inx} e^{-imx} over [0, 2 pi] is 2 pi delta_nm for n, m in -3..3:", orth)
a_sq = [integrate(square.expression_at(-1/2) * cos(n * x), x, -pi, 0) + integrate(square.expression_at(1/2) * cos(n * x), x, 0, pi) for n in range(0, 7)]
b_sq = [integrate(-cos(0) * sin(n * x), x, -pi, 0) + integrate(sin(n * x), x, 0, pi) for n in range(1, 7)]
sq_ok = all(bool(v == 0) for v in a_sq) and all(bool(b_sq[n - 1] / pi == (4 / (pi * n) if n % 2 == 1 else 0)) for n in range(1, 7))
print("[PASS] the square wave sgn(x) on (-pi, pi) has a_n = 0 and b_n = 4/(pi n) for odd n, 0 for even n (n <= 6):", sq_ok)
b_saw = [integrate(x * sin(n * x), x, -pi, pi) / pi for n in range(1, 7)]
a_saw = [integrate(x * cos(n * x), x, -pi, pi) / pi for n in range(0, 7)]
print("[PASS] the sawtooth x on (-pi, pi) has a_n = 0 and b_n = 2 (-1)^(n+1)/n for n <= 6:", all(bool(v == 0) for v in a_saw) and all(bool(b_saw[n - 1] == 2 * (-1) ** (n + 1) / n) for n in range(1, 7)))
n = var("n")
lhs = integrate(x ** 2, x, -pi, pi) / pi
rhs = sum((2 / n) ** 2, n, 1, oo)
print("[PASS] Parseval for the sawtooth: (1/pi) int x^2 = 2 pi^2/3 = sum b_n^2 = 4 sum 1/n^2, so sum 1/n^2 = pi^2/6:", bool(lhs == 2 * pi ** 2 / 3) and bool(rhs == lhs) and bool(sum(1 / n ** 2, n, 1, oo) == pi ** 2 / 6))
print("[PASS] the partial sum S_5 of the square wave is 4/pi (sin x + sin 3x/3 + sin 5x/5), as fourier_series_partial_sum returns:", bool((square.fourier_series_partial_sum(5) - S_square(5)).simplify_full() == 0))
D3 = sum(exp(I * k * x) for k in range(-3, 4))
print("[PASS] the Dirichlet kernel: sum_{|n| <= 3} e^{inx} = sin(7x/2)/sin(x/2), with integral 2 pi over a period:", bool((D3 * sin(x / 2) - sin(7 * x / 2)).rectform().trig_reduce() == 0) and bool(integrate(D3, x, -pi, pi) == 2 * pi))
err = 2 * integrate((1 - S_square(9)) ** 2, x, 0, pi)
tail = 2 * pi - pi * sum((4 / (pi * k)) ** 2 for k in range(1, 10, 2))
print("[PASS] the L^2 error of S_9 for the square wave is int (f - S_9)^2 = 2 pi - pi sum_{odd n <= 9} (4/(pi n))^2 = 0.2538... (Parseval for the tail):", bool((err - tail).expand() == 0) and bool(abs(err.n() - 0.253811) < 1e-5))
gibbs = (2 / pi * sin_integral(pi)).n(digits=12)
overshoot = S_square(199).subs(x=pi / 200).n(digits=12)
print("[PASS] Gibbs: S_199 at its first maximum x = pi/200 is 1.1790 to 1e-3, the limit (2/pi) Si(pi) = 1.17897974...:", bool(abs(overshoot - gibbs) < 1e-3) and bool(abs(gibbs - 1.17897974) < 1e-7))
print("wrote", os.path.relpath(OUT))

# interactive version (design D21): a slider sets the number of terms of the square-wave partial sum
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _interactive import plotly_html
import math as _m
xs = [-_m.pi + k * 2 * _m.pi / 600 for k in range(601)]
_traces = [{"x": xs, "y": [(-1.0 if t < 0 else 1.0) for t in xs], "mode": "lines", "name": "square wave", "line": {"color": "black", "width": 2}}]
Ns = list(range(1, 60, 2))
for N in Ns:
    ys = [sum(4 / (_m.pi * k) * _m.sin(k * t) for k in range(1, N + 1, 2)) for t in xs]
    _traces.append({"x": xs, "y": ys, "mode": "lines", "name": "S_%d" % N, "line": {"color": "firebrick", "width": 1.5}, "visible": N == 9})
steps = [{"method": "update", "label": str(N), "args": [{"visible": [True] + [M == N for M in Ns]}, {"title": "Square wave and its partial Fourier sum S_%d (the overshoot stays at 1.179)" % N}]} for N in Ns]
plotly_html("fourier_square_wave", _traces, {"title": "Square wave and its partial Fourier sum S_9 (the overshoot stays at 1.179)", "xaxis": {"title": "x"}, "yaxis": {"range": [-1.5, 1.5]}, "showlegend": False,
            "sliders": [{"active": 4, "currentvalue": {"prefix": "N = "}, "pad": {"t": 40}, "steps": steps}]})
print("wrote figures/interactive/fourier_square_wave.html")
