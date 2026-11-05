# Figure: exact anomalous dimensions of linear Dyson-Schwinger equations (Balduf, Examples 103 and 105): toy model
# -arcsin(pi a)/pi and the four-dimensional multiedge (sqrt(1 + 4a) - 1)/2, with series truncations.
# Output: figures/ren-dse-anomalous.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ren-dse-anomalous.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
a = np.linspace(0, 1 / np.pi, 300)
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(a, -np.arcsin(np.pi * a) / np.pi, color="black", lw=2, label=r"toy: $-\frac{1}{\pi}\arcsin(\pi\alpha)$")
ax.plot(a, -a, ":", color="gray", lw=1.2); ax.plot(a, -a - np.pi**2 * a**3 / 6, ":", color="black", lw=1.2)
ax.plot(a, (np.sqrt(1 + 4 * a) - 1) / 2, "--", color="royalblue", lw=2, label=r"multiedge: $\frac{\sqrt{1 + 4\alpha} - 1}{2}$")
ax.plot(a, a - a**2 + 2 * a**3, ":", color="royalblue", lw=1.2)
ax.axhline(0, color="gray", lw=0.6)
ax.set_xlabel(r"$\alpha$"); ax.set_ylabel(r"$\gamma(\alpha)$"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT)
var('al g')
print("[PASS] sin(pi g)/pi = -al is solved by g = -arcsin(pi al)/pi:", bool((sin(pi * (-arcsin(pi * al) / pi)) / pi + al).simplify_full() == 0))
print("[PASS] the toy series is -a - pi^2 a^3/6 - 3 pi^4 a^5/40:", bool(taylor(-arcsin(pi * al) / pi, al, 0, 5) == -al - pi^2 * al^3 / 6 - 3 * pi^4 * al^5 / 40))
print("[PASS] the multiedge solves g(1 + g) = a with Catalan coefficients 1, -1, 2, -5, 14:", bool(taylor((sqrt(1 + 4 * al) - 1) / 2, al, 0, 5) == al - al^2 + 2 * al^3 - 5 * al^4 + 14 * al^5))
print("wrote", os.path.relpath(OUT))
