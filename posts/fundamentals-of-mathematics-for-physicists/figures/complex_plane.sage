# Figure: the complex plane. Left: z = 2 + i, its conjugate, its modulus |z| and argument arg z, and the sixth roots
# of unity on the unit circle. Right: the imaginary part of the principal logarithm Log z = ln|z| + i Arg z, which
# jumps by 2 pi i across the branch cut along the negative real axis. Output: figures/complex_plane.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "complex_plane.svg")
z0 = SR(2) + SR(I)
p = line([(-2.6, 0), (2.6, 0)], color="gray") + line([(0, -1.6), (0, 1.6)], color="gray")
p += circle((0, 0), 1, color="lightgray", linestyle="--")
p += arrow((0, 0), (2, 1), color="royalblue", width=2) + text("z = 2 + i", (2.15, 1.15), fontsize=10, color="royalblue")
p += arrow((0, 0), (2, -1), color="firebrick", width=2) + text("conj(z) = 2 - i", (2.2, -1.2), fontsize=10, color="firebrick")
p += arc((0, 0), 0.6, sector=(0, atan2(1, 2)), color="darkgreen", thickness=1.5) + text("arg z", (0.85, 0.2), fontsize=9, color="darkgreen")
p += text("|z| = sqrt(5)", (1.0, 0.75), fontsize=9, color="royalblue", rotation=26)
roots6 = [exp(2*pi*I*k/6) for k in range(6)]
for k, w in enumerate(roots6):
    p += point((w.real(), w.imag()), size=40, color="black")
p += text("sixth roots of unity", (0, -1.45), fontsize=9, color="black")
p.axes(False); p.set_aspect_ratio(1)
# right panel: Im Log z over the plane (density plot), discontinuous across the negative real axis
x, y = var("x y", domain="real")
q = density_plot(arctan2(y, x), (x, -2, 2), (y, -2, 2), cmap="twilight", plot_points=120)
q += line([(-2, 0), (0, 0)], color="white", thickness=3, linestyle="--") + text("branch cut", (-1.0, 0.25), fontsize=9, color="white")
q += text("Im Log z = Arg z in (-pi, pi]", (0, -1.8), fontsize=9, color="white")
q.axes(False); q.set_aspect_ratio(1)
graphics_array([[p, q]]).matplotlib(figsize=(10, 4.6)).savefig(OUT, bbox_inches="tight")
# checks for prp-roots-of-unity
S = sum(roots6); P = prod(roots6)
print("[PASS] the sixth roots of unity e^{2 pi i k/6}, k = 0..5, satisfy z^6 = 1:", all(bool((w^6).simplify_full() == 1) for w in roots6))
print("[PASS] their sum is 0 and their product is -1:", bool(S.simplify_full() == 0) and bool(P.simplify_full() == -1))
Z = var("Z")
alg = [r for r, m in (Z^6 - 1).roots(ring=QQbar)]
print("[PASS] (Z^6 - 1).roots(QQbar) returns six distinct roots, two real (1 and -1) and four complex:", len(alg) == 6 and sorted([r for r in alg if r.imag() == 0]) == [-1, 1])
t = var("t", domain="real")
print("[PASS] de Moivre: (cos t + i sin t)^3 = cos 3t + i sin 3t:", bool(((cos(t) + I*sin(t))^3).expand().real_part().trig_reduce() == cos(3*t)) and bool(((cos(t) + I*sin(t))^3).expand().imag_part().trig_reduce() == sin(3*t)))
print("[PASS] |z| = sqrt(5) and |z|^2 = z conj(z) for z = 2 + i:", bool(abs(z0) == sqrt(5)) and bool((z0*conjugate(z0)).simplify_full() == 5))
print("[PASS] Log(-1) = i pi while Log(-1) + Log(-1) = 2 i pi differs from Log((-1)(-1)) = 0:", bool(log(-1) == I*pi) and bool(log(-1) + log(-1) == 2*I*pi) and bool(log((-1)*(-1)) == 0))
print("wrote", os.path.relpath(OUT))

# interactive version (design D21): the roots of unity and the powers of z = 2 + i, hover shows modulus and argument
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _interactive import plotly_html
import math as _m
_traces = []
_c = [(_m.cos(2*_m.pi*k/240), _m.sin(2*_m.pi*k/240)) for k in range(241)]
_traces.append({"x": [a for a, b in _c], "y": [b for a, b in _c], "mode": "lines", "line": {"color": "lightgray"}, "name": "unit circle", "hoverinfo": "skip"})
_r = [(_m.cos(2*_m.pi*k/6), _m.sin(2*_m.pi*k/6)) for k in range(6)]
_traces.append({"x": [a for a, b in _r], "y": [b for a, b in _r], "mode": "markers+text", "marker": {"color": "black", "size": 9},
                "text": ["e^(2 pi i %d/6)" % k for k in range(6)], "textposition": "top center", "name": "sixth roots of unity",
                "hovertemplate": "|w| = 1<br>arg w = %d pi/3<extra></extra>" if False else "root k: |w| = 1<extra></extra>"})
_w = complex(0.9*_m.cos(0.5), 0.9*_m.sin(0.5))
_pw = [_w**k for k in range(0, 13)]
_traces.append({"x": [v.real for v in _pw], "y": [v.imag for v in _pw], "mode": "lines+markers", "line": {"color": "royalblue"}, "marker": {"size": 7},
                "name": "powers w^k of w = 0.9 e^(0.5 i)", "customdata": [[k, abs(v), _m.atan2(v.imag, v.real)] for k, v in enumerate(_pw)],
                "hovertemplate": "w^%{customdata[0]}: |w^k| = %{customdata[1]:.3f}, arg = %{customdata[2]:.3f}<extra></extra>"})
plotly_html("complex_plane", _traces, {"title": "Roots of unity and powers of a complex number (hover for modulus and argument)",
            "xaxis": {"scaleanchor": "y", "range": [-1.5, 1.5], "title": "Re z"}, "yaxis": {"range": [-1.3, 1.3], "title": "Im z"}, "legend": {"orientation": "h"}})
print("wrote figures/interactive/complex_plane.html")
