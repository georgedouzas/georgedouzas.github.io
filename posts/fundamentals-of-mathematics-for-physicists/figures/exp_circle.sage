# Figure: the exponential map of the circle group. The Lie algebra u(1) = i R is the tangent line to U(1) at the
# identity 1; exp wraps the line onto the circle, i t -> e^{it}, with the segment [0, 2 pi) covering the circle once
# and 2 pi i, 4 pi i, ... all landing on 1: the kernel of exp is 2 pi i Z. Output: figures/exp_circle.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_circle.svg")
p = circle((0, 0), 1, color="black", thickness=1.5)
p += line([(-0.3, 2.4), (7.3, 2.4)], color="royalblue", thickness=1.5)
p += text(r"$U(1)$", (-1.35, 0.9), fontsize=11) + text(r"$\mathfrak{u}(1)=i\mathbb{R}$, the tangent line at $1$, parametrised by $t$", (3.5, 2.85), fontsize=10, color="royalblue")
ts = [0, 0.9, 1.8, 2.7, 3.6, 4.5, 5.4, 2*pi, 7.0]
for k, tk in enumerate(ts):
    col = hue(0.85 * k / (len(ts) - 1))
    p += point((float(tk), 2.4), color=col, size=45) + point((float(cos(tk)), float(sin(tk))), color=col, size=45)
p += text("$t=0$", (0, 2.1), fontsize=8) + text("$t=2\pi$", (float(2*pi), 2.1), fontsize=8)
p += arrow((3.5, 1.9), (1.6, 0.9), color="firebrick", width=1.2, arrowsize=2) + text(r"$\exp: it\mapsto e^{it}$ (same colour, same point)", (4.4, 1.35), fontsize=9, color="firebrick")
p += text(r"$t$ and $t+2\pi$ land on the same point: $\ker\exp = 2\pi i\mathbb{Z}$", (3.5, -0.6), fontsize=9)
p += text("$1$", (1.22, -0.18), fontsize=10)
p.axes(False); p.set_aspect_ratio(1)
p.save(OUT, figsize=[7.5, 3.6])
t = var("t", domain="real")
print("[PASS] exp(i t) has modulus 1 and exp(2 pi i) = 1:", bool(abs(exp(I * t)).simplify_full() == 1) and bool(exp(2 * pi * I) == 1))
print("[PASS] exp is a homomorphism (R, +) -> U(1): e^{i(s+t)} = e^{is} e^{it}:", bool((exp(I * (var("s") + t)) - exp(I * var("s")) * exp(I * t)).simplify_full() == 0))
print("wrote", os.path.relpath(OUT))
