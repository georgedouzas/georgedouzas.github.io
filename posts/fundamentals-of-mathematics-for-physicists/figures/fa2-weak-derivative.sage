# Figure: |x| with its weak derivative sgn x, and the Cantor function. Output: figures/fa2-weak-derivative.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fa2-weak-derivative.svg")
x = var("x")
L = plot(abs(x), (-1, 1), color="royalblue", thickness=1.8, legend_label=r"$|x|$")
L += line([(-1, -1), (0, -1)], color="darkorange", thickness=1.8, legend_label=r"$\operatorname{sgn} x$") + line([(0, 1), (1, 1)], color="darkorange", thickness=1.8)
L.set_legend_options(loc="lower right", font_size=9)
def cantor(t, depth=30):
    s, scale = 0.0, 0.5
    for _ in range(depth):
        if t < 1 / 3:
            t = 3 * t
        elif t > 2 / 3:
            s += scale; t = 3 * t - 2
        else:
            return s + scale
        scale /= 2
    return s
R = line([(k / 3000.0, cantor(k / 3000.0)) for k in range(3001)], color="firebrick", thickness=1.4, legend_label="Cantor function")
R.set_legend_options(loc="upper left", font_size=9)
graphics_array([L, R]).save(OUT, figsize=[10, 4])
phi = exp(-1 / (1 - x ^ 2)) * (x + 1 / 3)
lhs = numerical_integral(abs(x) * diff(phi, x), -1, 1)[0]
rhs = -numerical_integral(sign(x) * phi, -1, 1)[0]
print("[PASS] int |x| phi' = -int sgn(x) phi for a bump test function phi:", abs(lhs - rhs) < 1e-9)
print("[PASS] the Cantor function: c(0) = 0, c(1) = 1, c = 1/2 on [1/3, 2/3], c(1/4) = 1/3:", cantor(0.0) == 0 and abs(cantor(0.999999999999) - 1) < 1e-6 and cantor(0.5) == 0.5 and abs(cantor(0.25) - 1 / 3) < 1e-8)
print("[PASS] the weak derivative of x|x| is 2|x|:", abs(numerical_integral(x * abs(x) * diff(phi, x), -1, 1)[0] + numerical_integral(2 * abs(x) * phi, -1, 1)[0]) < 1e-9)
print("wrote", os.path.relpath(OUT))
