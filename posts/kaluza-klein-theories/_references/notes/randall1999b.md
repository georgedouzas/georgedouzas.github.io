# Identity

- arXiv: hep-th/9906064 (v1, 8 Jun 1999). Preprint numbers MIT-CTP-2874, PUPT-1867, BUHEP-99-13.
- Title: "An Alternative to Compactification".
- Authors: Lisa Randall (Princeton / MIT CTP), Raman Sundrum (Boston University).
- Year: 1999. Venue: not stated in the PDF (preprint; the published version is Phys. Rev. Lett. 83 (1999) 4690, not stated in the file).
- PDF is 10 pages; printed page p = PDF page p+1 (title/abstract page is unnumbered). Commonly called "RS2".

# Structure

The paper has a single numbered heading, "1 Introduction" (PDF p. 2); everything after it is unsectioned running text. The outline below is by content.

1. Abstract — PDF p. 1 (unnumbered). Newton's law does not imply only four non-compact dimensions; a single 3-brane in 5D; no gap in the KK spectrum yet 4D gravity reproduced.
2. Section 1, Introduction — PDF pp. 2-3 (printed 1-2). Factorizable vs non-factorizable geometry; graviton bound state; wave equation eqs (1)-(2) as an analogue quantum-mechanics problem.
3. Set-up and background solution — PDF pp. 3-4 (printed 2-3). Single positive-tension brane plus a regulator brane at distance $\pi r_c$; action eq (3); solution eq (4); tuning eq (5); labels of visible/hidden branes reversed relative to RS1; effective action eq (6) and $M_{Pl}^2$ eq (7), finite as $r_c \to \infty$.
4. KK reduction of tensor fluctuations — PDF pp. 4-5 (printed 3-4). Mode equation eq (8); change of variables to $z$; Schrodinger form eqs (9)-(10) ("volcano" potential); zero mode $\hat\psi_0$ (footnote 1); Bessel-function continuum eqs (11)-(13); regulator-brane boundary condition eq (14).
5. Density of states and Newton's law — PDF pp. 5-6 (printed 4-5). Normalisation $N_m$, measure $dm$, TeV-scale KK states of RS1 recovered; extension to a fully infinite dimension; potential eqs (15)-(16).
6. Consistency: KK production, strong coupling at large $z$ — PDF pp. 6-8 (printed 5-7). Suppression $(p/k)^2$; graviton self-coupling eq (17); blue-shift near the AdS horizon; probability $10^{-18}$ estimate.
7. Conclusions and outlook (unlabelled) — PDF pp. 8-9 (printed 7-8). Cosmological-constant-like tuning; moduli-problem interpretation; difference from compactified theories; speculation on a dual cut-off CFT. Acknowledgements at the end of PDF p. 9.
8. References [1]-[13] — PDF p. 10 (printed 9).

# Conventions

- Natural units; $M_{Pl}$ the 4D Planck scale, $M$ the 5D fundamental scale, $k$ the AdS curvature scale ("of order the Planck scale"), $G_N$ Newton's constant.
- Signature: not stated explicitly. The solution eq (4), $ds^2 = e^{-2k|y|}\eta_{\mu\nu}dx^\mu dx^\nu + dy^2$, with a spacelike fifth direction requires mostly-plus $\eta_{\mu\nu}$, i.e. $(-,+,+,+,+)$.
- Coordinates: $x^\mu$, $\mu = 0,\dots,3$; the fifth coordinate is the proper-distance coordinate $y = r_c\phi$ ("the new coordinate $y$ is $r_c\phi$ in the coordinates of Ref. [8]", PDF p. 3), with $0 \le y \le \pi r_c$ while the regulator brane is present; orbifold boundary conditions (only even functions of $y$) as in RS1; later the set-up is extended to $y \in \mathbb{R}$ with both even and odd modes (PDF p. 6). Conformal-type coordinate $z \equiv \mathrm{sgn}(y)(e^{k|y|}-1)/k$ (PDF p. 4).
- Action normalisation: $S_{gravity} = \int d^4x\int dy\,\sqrt{-G}\{-\Lambda + 2M^3R\}$, eq (3), the same $2M^3$ convention as RS1 (equivalently $1/(2\kappa_5^2) = 2M^3$). The $y$-integration range is not written in eq (3); eqs (6)-(7) integrate over $0 \le y \le \pi r_c$ with an explicit factor 2.
- Brane tensions: eq (3) prints $S_{brane} = \int d^4x\sqrt{-g_{brane}}\{V_{brane} + \mathcal{L}_{brane}\}$, i.e. with a plus sign in front of $V_{brane}$, opposite to RS1's $\{\mathcal{L} - V\}$ (see Caveats). The physical brane at $y=0$ is "a single 3-brane with positive tension" (PDF p. 3); $V_{brane} = +24M^3k$, the regulator brane $V_{brane'} = -24M^3k$, eq (5). The physical (positive-tension) brane is now the "visible" brane: "we have reversed the labels for the 'visible' and 'hidden' branes relative to Ref. [8]" (PDF p. 4).
- Bulk: $\Lambda = -24M^3k^2 < 0$, "a slice of $AdS_5$" (PDF p. 3).
- Gauge for fluctuations: $\partial^\mu h_{\mu\nu} = h^\mu_\mu = 0$ (PDF p. 4); tensor indices omitted on $\psi$, $\hat\psi$.

# Equations the tutorial uses

(a) Action (PDF p. 3, printed p. 2), eq (3), exactly as printed:
$$S = S_{gravity} + S_{brane} + S_{brane'},\qquad S_{gravity} = \int d^4x\int dy\,\sqrt{-G}\{-\Lambda + 2M^3R\},\qquad S_{brane} = \int d^4x\sqrt{-g_{brane}}\{V_{brane} + \mathcal{L}_{brane}\}. \qquad \text{(3)}$$
"$\Lambda$ and $V_{brane}$ are cosmological terms in the bulk and boundary respectively."

(b) Background solution (PDF p. 3), eq (4), "derived in Ref. [8]":
$$ds^2 = e^{-2k|y|}\eta_{\mu\nu}dx^\mu dx^\nu + dy^2, \qquad \text{(4)}$$
"where $0 \le y \le \pi r_c$ is the extra-dimensional coordinate and $r_c$ is essentially a compactification 'radius'."

(c) Tuning conditions (PDF p. 3), eq (5):
$$V_{brane} = -V_{brane'} = 24M^3k, \qquad \Lambda = -24M^3k^2, \qquad \text{(5)}$$
"this condition amounts to setting the cosmological constant of the four-dimensional world to zero in this context, and we simply accept this necessary fine tuning without further explanation here." The Einstein equations are not re-derived; $\sigma = k|y|$ and $k^2 = -\Lambda/(24M^3)$ are inherited from RS1 (randall1999a, eqs (7)-(11)).

(d) 4D Planck mass (PDF p. 4, printed p. 3), eqs (6)-(7), as printed:
$$S_{eff} \supset \int d^4x\int_0^{\pi r_c}dy\;2M^3 r_c\,e^{-2k|y|}\sqrt{-\bar g}\,\bar R, \qquad \text{(6)}$$
$$M_{Pl}^2 = 2M^3\int_0^{\pi r_c}dy\,e^{-2k|y|} = \frac{M^3}{k}\big[1 - e^{-2kr_c\pi}\big]. \qquad \text{(7)}$$
"We see that there is a well-defined value for $M_{Pl}$, even in the $r_c \to \infty$ limit. This is a clue that one can get a sensible effective four-dimensional theory, with the usual Newtonian force law, even in the infinite radius limit, in sharp contrast to the product-space expectation that $M_{Pl}^2 = M^3 r_c\pi$." (The factor $r_c$ in eq (6) is a misprint carried over from the $\phi$-coordinate of RS1; eq (7) is dimensionally correct without it.)

(e) Graviton zero mode and continuum (PDF pp. 4-5, printed pp. 3-4). For $G_{\mu\nu} = e^{-2k|y|}\eta_{\mu\nu} + h_{\mu\nu}(x,y)$, $h(x,y) = \psi(y)e^{ip\cdot x}$, $p^2 = m^2$, the linearised equation is eq (8):
$$\Big[\frac{-m^2}{2}e^{2k|y|} - \frac{1}{2}\partial_y^2 - 2k\,\delta(y) + 2k^2\Big]\psi(y) = 0. \qquad \text{(8)}$$
With $z \equiv \mathrm{sgn}(y)\,(e^{k|y|}-1)/k$, $\hat\psi(z) \equiv \psi(y)e^{k|y|/2}$, $\hat h(x,z) \equiv h(x,y)e^{k|y|/2}$, this becomes the analogue Schrodinger problem, eqs (9)-(10):
$$\Big[-\frac{1}{2}\partial_z^2 + V(z)\Big]\hat\psi(z) = m^2\hat\psi, \qquad V(z) = \frac{15k^2}{8(k|z|+1)^2} - \frac{3k}{2}\delta(z). \qquad \text{(9), (10)}$$
"The $\delta$-function supports a single normalizable bound state mode; the remaining eigenstates are continuum modes"; "since the potential falls off to zero as $|z| \to \infty$, there is no gap, and the continuum modes asymptote to plane waves"; "the amplitudes of the continuum modes is suppressed near the origin, due to the potential barrier near $z=0$"; "the continuum KK states have all possible $m^2 > 0$." Zero mode (text, no equation number): $\hat\psi_0(z) = k^{-1}(k|z|+1)^{-3/2}$. Continuum modes are linear combinations of $(|z|+1/k)^{1/2}Y_2(m(|z|+1/k))$ and $(|z|+1/k)^{1/2}J_2(m(|z|+1/k))$. Small-argument forms, eq (11):
$$J_2(m(|z|+1/k)) \sim \frac{m^2(|z|+1/k)^2}{8},\qquad Y_2(m(|z|+1/k)) \sim -\frac{4}{\pi m^2(|z|+1/k)^2} - \frac{1}{\pi}. \qquad \text{(11)}$$
Boundary condition at $z=0$ selects, for small $m$, eq (12):
$$\hat\psi_m \sim N_m(|z|+1/k)^{1/2}\Big[Y_2(m(|z|+1/k)) + \frac{4k^2}{\pi m^2}J_2(m(|z|+1/k))\Big]. \qquad \text{(12)}$$
Large-$mz$ asymptotics, eq (13): $\sqrt z J_2(mz) \sim \sqrt{2/(\pi m)}\cos(mz - 5\pi/4)$, $\sqrt z Y_2(mz) \sim \sqrt{2/(\pi m)}\sin(mz - 5\pi/4)$. Regulator brane at $z_c \equiv (e^{k\pi r_c}-1)/k$ imposes eq (14): $\partial_z\hat\psi(z_c) = -\frac{3k}{2(kz_c+1)}\hat\psi(z_c)$, which quantises $m$ in units of $1/z_c$ with $N_m \sim \pi m^{5/2}/(4k^2\sqrt{z_c})$ (PDF p. 6); dropping $1/\sqrt{z_c}$ gives the continuum measure "simply $dm$".

(f) Newton's law (PDF p. 6, printed p. 5), eqs (15)-(16), exactly as printed:
$$V(r) \sim G_N\frac{m_1m_2}{r} + \int_0^\infty dm\,\frac{G_N}{k}\frac{m_1m_2e^{-mr}}{r}\frac{m}{k}. \qquad \text{(15)}$$
$$V(r) = G_N\frac{m_1m_2}{r}\Big(1 + \frac{1}{r^2k^2}\Big). \qquad \text{(16)}$$
"The coupling $G_N/k$ in the second term is nothing but the fundamental coupling of gravity, $1/M^3$, by Eq. (7)"; "the extra power of $m/k$ arises from the suppression of continuum wavefunctions at $z=0$". Check: $\int_0^\infty dm\,m\,e^{-mr} = 1/r^2$, so the second term of (15) is $G_N m_1m_2/(k^2r^3)$, reproducing (16).

(g) Why the extra dimension can be infinite (PDF pp. 2-3, 4, 6-7): (i) a curved background supports a normalizable bound-state graviton zero mode ("although space is indeed infinite in extent, the graviton is confined to a small region within this space", PDF p. 2); (ii) $M_{Pl}^2$ in eq (7) stays finite as $r_c \to \infty$ because of the warp factor, "$M_{Pl}$ is determined by the higher-dimensional curvature rather than the size of the extra dimension" (PDF p. 2); (iii) the gapless continuum couples weakly, giving only the $1/(r^2k^2)$ correction in (16) and KK production suppressed by $(p/k)^2$ relative to the zero mode (PDF p. 7). Graviton self-coupling, eq (17): $S \supset \int d^4x\int dz\,z^{3/2}\,\hat h^{\mu\nu}\partial_\mu\hat h_{\alpha\beta}\partial_\nu\hat h^{\alpha\beta}/M_{Pl}$, which grows at large $z$ (AdS horizon blue-shift).

# Attributable statements

1. "Conventional wisdom states that Newton's force law implies only four non-compact dimensions. We demonstrate that this is not necessarily true in the presence of a non-factorizable background geometry." (Abstract, PDF p. 1)
2. "The previous properties rely on a factorizable geometry, namely the metric of the four familiar dimensions is independent of coordinate in the extra dimensions. The story can change significantly when this assumption is dropped." (Sec. 1, PDF p. 2)
3. "a curved background can support a 'bound state' of the higher-dimensional graviton, which is localized in the extra dimensions." (Sec. 1, PDF p. 2)
4. "If there is a zero-mode (which is guaranteed if the background preserves four-dimensional Poincare-invariance) which is also a normalizable state in the spectrum of Eq. (2), it is the wave function associated with the four-dimensional graviton." (PDF p. 3)
5. "in our theory, there is a continuous KK spectrum with no gap. However, four-dimensional physics is extremely well approximated because the bound state mode reproduces conventional four-dimensional gravity, while the other KK modes give only a small correction." (PDF p. 3)
6. "the KK states couple $10^{15}$ more strongly to matter on the brane at $z_c$ than does the massless graviton" — proving the RS1 claim of TeV-coupled KK gravitons (PDF p. 6).
7. "The leading term due to the bound state mode is the usual Newtonian potential; the KK modes generate an extremely suppressed correction term, for $k$ taking the expected value of order the fundamental Planck scale and $r$ of the size tested with gravity." (PDF p. 6)
8. "The scenario consists of a single 3-brane, (a piece of) $AdS_5$ in the bulk, and an appropriately tuned tension on the brane. The need for this delicate adjustment is the equivalent of the cosmological constant problem in this context, and is taken as a given and not solved." (PDF p. 8)
9. "one interpretation of our result is as a solution to the moduli problem, for the particular modulus determining the distance between two branes. It says that the usual disasterous possibility, namely that the modulus runs away to infinity, is perfectly acceptable." (PDF p. 9)
10. "the extra-dimensional measure is straightforwardly $dy$ in the coordinate $y$, which takes values on the entire real line. The low dimensional Planck scale and all physical parameters of the effective four-dimensional theory are independent of $r_c$, so long as it is much greater than $1/k$." (PDF p. 9)

# Caveats

- Sign of the brane term: eq (3) prints $\{V_{brane} + \mathcal{L}_{brane}\}$ whereas RS1 eq (4) prints $\{\mathcal{L} - V\}$; with eq (5) ($V_{brane} = +24M^3k$) and the text's "single 3-brane with positive tension", the physically consistent reading is RS1's sign (positive energy density $24M^3k$ on the $y=0$ brane). The tutorial should use the RS1 convention and note the discrepancy rather than copy eq (3) verbatim.
- Fine tuning: eq (5) is explicitly called a "necessary fine tuning" equivalent to setting the 4D cosmological constant to zero (PDF p. 3) and "the equivalent of the cosmological constant problem" (PDF p. 8). Not addressed.
- Radius stabilisation: not needed here — $r_c \to \infty$ is taken, "the modulus is not coupled to matter on the brane, and the need for a modulus mass is eliminated" (PDF p. 9). The regulator brane' (negative tension) is a device to define the mode normalisation and is then removed.
- Jump conditions: encoded in the $\delta(y)$ term of eq (8) (coefficient $-2k$) and the $-(3k/2)\delta(z)$ term of eq (10); the background jump conditions themselves are not re-derived but taken from RS1 (eqs (8), (10)-(11) there). Orbifold boundary conditions (even $\psi$) are used first; later both parities are allowed on the full line (PDF p. 6).
- Eq (6) contains a stray $r_c$ (misprint); eq (7) is the correct result. The $M_{Pl}$ normalisation is the same "$2M^{D-2}R$" convention as RS1: matching to $2M_{Pl}^2\bar R$ gives eq (7); with $(M_{Pl}^2/2)\bar R$ one gets $4M^3/k$ instead of $M^3/k$. The result $M_{Pl}^2 = M^3/k$ (for $r_c \to \infty$) and $G_N/k = 1/M^3$ (PDF p. 6) are stated in this convention.
- Eq (15) is written with "$\sim$"; eq (16) with "=" but is the leading-order estimate (normalisations of order one not tracked). The $1/(r^2k^2)$ correction is the accepted RS2 result at this order; later literature refines the coefficient (e.g. a factor $2/3$ in some normalisations), which this paper does not compute.
- Validity: the linearised analysis breaks down at large $z$ (strong coupling near the AdS horizon, eq (17)); the authors argue low-energy brane processes reach $z > 10^9$ with probability $\sim 10^{-18}$ and that the strong-coupling region presumably has a string-theoretic description (PDF pp. 7-8). The theory "clearly breaks down before $r_c \to \infty$ for (rare) processes involving Kaluza-Klein modes" (PDF p. 9).
- No explicit $y$ range appears in eq (3); the Introduction uses generic extra coordinates $z_j$ in eqs (1)-(2), unrelated to the later specific $z(y)$.
- Reversed brane labels relative to RS1: here the positive-tension brane at $y=0$ is the physical/visible brane; in RS1 it was the hidden brane. Both solutions are the same metric.
