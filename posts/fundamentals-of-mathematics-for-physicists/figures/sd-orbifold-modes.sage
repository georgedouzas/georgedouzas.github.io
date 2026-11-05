# Figure: modes on the orbifold S^1/Z_2 (y ~ -y, fundamental domain [0, pi R], R = 1): even modes cos(n y) and odd modes
# sin(n y); a spinor with psi(-y) = gamma_5 psi(y) has its right-handed part even and its left-handed part odd, so only the
# right-handed part has a zero mode. Output: figures/sd-orbifold-modes.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sd-orbifold-modes.svg")
y = var("y")
cols = ["black", "royalblue", "darkgreen", "darkorange"]
P = Graphics()
for n in range(0, 4):
    P += plot(cos(n * y), (y, 0, pi), color=cols[n], thickness=2 if n == 0 else 1.2, legend_label=r"even: $\cos %dy$" % n)
    if n > 0:
        P += plot(sin(n * y), (y, 0, pi), color=cols[n], linestyle="--", legend_label=r"odd: $\sin %dy$" % n)
P.axes_labels([r"$y$", ""]); P.set_legend_options(loc="lower left", font_size=7, ncol=2)
P.save(OUT, figsize=[7, 3.8], ymin=-1.6, ymax=1.2, ticks=[pi / 4, 0.5], tick_formatter=[pi, None])
print("[PASS] cos(n y) is even and sin(n y) odd under y -> -y:", all(bool(cos(n * (-y)) == cos(n * y)) and bool(sin(n * (-y)) == -sin(n * y)) for n in range(4)))
print("[PASS] the only y-independent mode is even: sin(0 y) = 0:", bool(sin(0 * y) == 0) and bool(cos(0 * y) == 1))
print("[PASS] odd modes vanish at the fixed points y = 0, pi:", all(bool(sin(n * yy) == 0) for n in range(1, 4) for yy in (0, pi)))
print("wrote", os.path.relpath(OUT))
