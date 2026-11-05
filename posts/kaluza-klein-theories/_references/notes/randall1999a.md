# Identity

- arXiv: hep-ph/9905221 (v1, 4 May 1999). Preprint numbers MIT-CTP-2860, PUPT-1860, BUHEP-99-9. (The running head on the title page reads "hep-th/9905221"; the arXiv stamp in the margin reads "hep-ph/9905221v1". The correct identifier is hep-ph/9905221.)
- Title: "A Large Mass Hierarchy from a Small Extra Dimension".
- Authors: Lisa Randall (Princeton / MIT CTP), Raman Sundrum (Boston University).
- Year: 1999. Venue: not stated in the PDF (preprint; the published version is Phys. Rev. Lett. 83 (1999) 3370, not stated in the file).
- PDF is 9 pages; printed page p = PDF page p+1 (title/abstract page is unnumbered). Commonly called "RS1".

# Structure

1. Abstract — PDF p. 1 (unnumbered). Warp factor from a slice of $AdS_5$; two 3-branes; KK gravitons at the weak scale with weak-scale couplings.
2. Section 1, Introduction — PDF pp. 2-3 (printed 1-2). Eq (1) (product-space Planck relation), eq (2) (the warped metric with $\sigma = k r_c \phi$ already inserted); contrast with large extra dimensions (ADD).
3. Section 2, The Set-Up — PDF p. 3 (printed 2). Orbifold $S^1/\mathbf{Z}_2$, induced metrics eq (3), the action eq (4).
4. Section 3, Classical Solution — PDF pp. 4-5 (printed 3-4). Einstein equations eq (5), ansatz eq (6), reduced equations eqs (7)-(8), solution eq (9), jump equation eq (10), tuning conditions eq (11), final metric eq (12). Footnote 1 on supersymmetric compatibility.
5. Section 4, Physical Implications — PDF pp. 5-8 (printed 4-7). Zero-mode ansatz eqs (13)-(14), effective action eq (15), 4D Planck mass eq (16), Higgs example eqs (17)-(20), general warped mass eq (21); modulus $T(x)$ and stabilisation; TeV-scale KK gravitons; change of viewpoint $x^\mu \to e^{k r_c \pi} x^\mu$ (footnote 2); collider phenomenology.
6. Section 5, Conclusions — PDF p. 8 (printed 7). Acknowledgements at the end of the same page.
7. References [1]-[10] — PDF p. 9 (printed 8).

# Conventions

- Natural units. $M_{Pl} = 2\times 10^{18}$ GeV is called the "effective four-dimensional (reduced) Planck scale" (PDF p. 2); $M$ is the fundamental 5D Planck scale; $k$ a curvature scale "of order the Planck scale" with $k < M$ assumed (PDF p. 5); $r_c$ the "compactification radius"; $\mu_c \equiv 1/r_c$.
- Signature: not stated explicitly. The ansatz eq (6), $ds^2 = e^{-2\sigma(\phi)}\eta_{\mu\nu}dx^\mu dx^\nu + r_c^2 d\phi^2$, with a spacelike fifth direction requires $\eta_{\mu\nu}$ mostly-plus, i.e. $(-,+,+,+)$ and 5D $(-,+,+,+,+)$.
- Coordinates: $x^\mu$, $\mu = 0,\dots,3$; the fifth coordinate is the angle $\phi \in [-\pi,\pi]$ with $(x,\phi) \sim (x,-\phi)$ (orbifold $S^1/\mathbf{Z}_2$), so the physical range is $0 \le \phi \le \pi$ and the physical length of the interval is $\pi r_c$ (PDF pp. 3-4). 5D indices $M,N = \mu,\phi$; 5D metric $G_{MN}$.
- Orbifold fixed points $\phi = 0$ (hidden brane, metric $g^{hid}_{\mu\nu}$) and $\phi = \pi$ (visible brane, metric $g^{vis}_{\mu\nu}$), eq (3).
- Action normalisation: $S_{gravity} = \int d^4x\int_{-\pi}^{\pi} d\phi\,\sqrt{-G}\,\{-\Lambda + 2M^3 R\}$, eq (4). So the Einstein-Hilbert coefficient is $2M^3$ (not $M^3$ nor $M^3/2$); in $1/(2\kappa_5^2)$ language, $1/(2\kappa_5^2) = 2M^3$, i.e. $\kappa_5^2 = 1/(4M^3)$, which is exactly the prefactor $1/(4M^3)$ in eq (5). Integration is over the full circle $-\pi \le \phi \le \pi$.
- Brane tensions: each brane Lagrangian is written $\mathcal{L} - V$, eq (4), so $V > 0$ is positive energy density (positive tension). The solution requires $V_{hid} = +24M^3k$ (hidden brane at $\phi=0$, positive tension) and $V_{vis} = -24M^3k$ (visible brane at $\phi=\pi$, negative tension), eq (11).
- Bulk cosmological constant $\Lambda < 0$ (AdS$_5$), $\Lambda = -24M^3k^2$, eq (11).
- 4D effective Planck mass is extracted by matching to $2M_{Pl}^2\sqrt{-\bar g}\,\bar R$, i.e. with the same "$2M^{D-2}$" normalisation as in 5D (see Caveats).

# Equations the tutorial uses

(a) The action (Sec. 2, PDF p. 3, printed p. 2), eq (4), exactly as printed:
$$S = S_{gravity} + S_{vis} + S_{hid},$$
$$S_{gravity} = \int d^4x \int_{-\pi}^{\pi} d\phi\,\sqrt{-G}\,\{-\Lambda + 2M^3 R\},$$
$$S_{vis} = \int d^4x\,\sqrt{-g_{vis}}\,\{\mathcal{L}_{vis} - V_{vis}\},\qquad S_{hid} = \int d^4x\,\sqrt{-g_{hid}}\,\{\mathcal{L}_{hid} - V_{hid}\}. \qquad \text{(4)}$$
with the induced metrics, eq (3): $g^{vis}_{\mu\nu}(x^\mu) \equiv G_{\mu\nu}(x^\mu,\phi=\pi)$, $g^{hid}_{\mu\nu}(x^\mu) \equiv G_{\mu\nu}(x^\mu,\phi=0)$. "From each 3-brane Lagrangian we have separated out a constant 'vacuum energy' which acts as a gravitational source even in the absence of particle excitations" (PDF p. 3).

(b) The warped ansatz (Sec. 3, PDF p. 4, printed p. 3), eq (6):
$$ds^2 = e^{-2\sigma(\phi)}\eta_{\mu\nu}dx^\mu dx^\nu + r_c^2 d\phi^2. \qquad \text{(6)}$$
"$r_c$ is the 'compactification radius' of the extra dimensional circle prior to orbifolding. After orbifolding, the size of the extra dimensional interval is $\pi r_c$." The Introduction's eq (2) is the same metric with the solution inserted: $ds^2 = e^{-2kr_c\phi}\eta_{\mu\nu}dx^\mu dx^\nu + r_c^2 d\phi^2$ (PDF p. 2; note eq (2) writes $\phi$, eq (12) writes $|\phi|$).

(c) Einstein equations and solution (Sec. 3, PDF pp. 4-5, printed pp. 3-4). The 5D Einstein equations, eq (5):
$$\sqrt{-G}\Big(R_{MN} - \tfrac{1}{2}G_{MN}R\Big) = -\frac{1}{4M^3}\Big[\Lambda\sqrt{-G}\,G_{MN} + V_{vis}\sqrt{-g_{vis}}\,g^{vis}_{\mu\nu}\,\delta^\mu_M\delta^\nu_N\,\delta(\phi-\pi) + V_{hid}\sqrt{-g_{hid}}\,g^{hid}_{\mu\nu}\,\delta^\mu_M\delta^\nu_N\,\delta(\phi)\Big]. \qquad \text{(5)}$$
With the ansatz they reduce to
$$\frac{6\sigma'^2}{r_c^2} = \frac{-\Lambda}{4M^3}, \qquad \text{(7)}$$
$$\frac{3\sigma''}{r_c^2} = \frac{V_{hid}}{4M^3 r_c}\,\delta(\phi) + \frac{V_{vis}}{4M^3 r_c}\,\delta(\phi-\pi). \qquad \text{(8)}$$
The solution of (7) "consistent with the orbifold symmetry $\phi \to -\phi$" is
$$\sigma = r_c|\phi|\sqrt{\frac{-\Lambda}{24M^3}}, \qquad \text{(9)}$$
(additive constant dropped; "this solution only makes sense if $\Lambda < 0$"; the bulk is "a slice of an $AdS_5$ geometry"). Treating the metric as periodic in $\phi$, eq (9) implies the jump equation
$$\sigma'' = 2r_c\sqrt{\frac{-\Lambda}{24M^3}}\,[\delta(\phi) - \delta(\phi-\pi)]. \qquad \text{(10)}$$
Comparing (10) with (8), a solution exists "only if $V_{hid}, V_{vis}, \Lambda$ are related in terms of a single scale $k$":
$$V_{hid} = -V_{vis} = 24M^3 k, \qquad \Lambda = -24M^3k^2. \qquad \text{(11)}$$
Equivalently $k^2 = -\Lambda/(24M^3)$ and $\sigma = k r_c|\phi|$. The final bulk metric is eq (12):
$$ds^2 = e^{-2kr_c|\phi|}\eta_{\mu\nu}dx^\mu dx^\nu + r_c^2 d\phi^2. \qquad \text{(12)}$$
"The compactification radius $r_c$ is effectively an arbitrary integration constant for this solution" (PDF p. 5).
Check: (9) in (7): $6k^2 r_c^2/r_c^2 = 6k^2 = -\Lambda/(4M^3)$ gives $\Lambda = -24M^3k^2$; (10) in (8): $3\cdot 2kr_c/r_c^2 = 6k/r_c = V_{hid}/(4M^3r_c)$ gives $V_{hid} = 24M^3k$, and the $-\delta(\phi-\pi)$ term gives $V_{vis} = -24M^3k$. Consistent.

(d) The mass hierarchy (Sec. 4, PDF pp. 5-6, printed pp. 4-5). Zero-mode ansatz, eqs (13)-(14):
$$ds^2 = e^{-2kT(x)|\phi|}[\eta_{\mu\nu} + \bar h_{\mu\nu}(x)]dx^\mu dx^\nu + T^2(x)d\phi^2, \qquad \bar g_{\mu\nu}(x) \equiv \eta_{\mu\nu} + \bar h_{\mu\nu}(x), \qquad \text{(13), (14)}$$
with $r_c = \langle T \rangle$; then $T$ is replaced by $r_c$. Curvature term of the effective action, eq (15):
$$S_{eff} \supset \int d^4x\int_{-\pi}^{\pi} d\phi\;2M^3 r_c\,e^{-2kr_c|\phi|}\sqrt{-\bar g}\,\bar R, \qquad \text{(15)}$$
and the 4D Planck mass, eq (16):
$$M_{Pl}^2 = M^3 r_c\int_{-\pi}^{\pi} d\phi\,e^{-2kr_c|\phi|} = \frac{M^3}{k}\big[1 - e^{-2kr_c\pi}\big]. \qquad \text{(16)}$$
"$M_{Pl}$ depends only weakly on $r_c$ in the large $kr_c$ limit." Visible-brane Higgs, eqs (17)-(19): since $g^{vis}_{\mu\nu} = e^{-2kr_c\pi}\bar g_{\mu\nu}$,
$$S_{vis} \supset \int d^4x\sqrt{-g_{vis}}\{g_{vis}^{\mu\nu}D_\mu H^\dagger D_\nu H - \lambda(|H|^2 - v_0^2)^2\} \qquad \text{(17)}$$
$$= \int d^4x\sqrt{-\bar g}\,e^{-4kr_c\pi}\{\bar g^{\mu\nu}e^{2kr_c\pi}D_\mu H^\dagger D_\nu H - \lambda(|H|^2 - v_0^2)^2\}, \qquad \text{(18)}$$
and after $H \to e^{kr_c\pi}H$,
$$S_{eff} \supset \int d^4x\sqrt{-\bar g}\{\bar g^{\mu\nu}D_\mu H^\dagger D_\nu H - \lambda(|H|^2 - e^{-2kr_c\pi}v_0^2)^2\}. \qquad \text{(19)}$$
Hence $v \equiv e^{-kr_c\pi}v_0$, eq (20), and in general, eq (21):
$$m \equiv e^{-kr_c\pi}m_0, \qquad \text{(21)}$$
"any mass parameter $m_0$ on the visible 3-brane in the fundamental higher-dimensional theory will correspond to a physical mass" $m$ "when measured with the metric $\bar g_{\mu\nu}$". "If $e^{kr_c\pi}$ is of order $10^{15}$, this mechanism produces TeV physical mass scales from fundamental mass parameters not far from the Planck scale"; "we only require $kr_c \approx 50$" (PDF p. 7).

(e) Change of viewpoint (PDF p. 7, no equation number): under $x^\mu \to e^{kr_c\pi}x^\mu$ the warp factor at $\phi=\pi$ is unity and at $\phi=0$ is $e^{2kr_c\pi}$; from the visible-brane viewpoint the TeV scale is fundamental and "the large Planck scale (the weakness of gravity) arises because of the small overlap of the graviton wave function in the fifth dimension (which is the warp factor) with our brane."

# Attributable statements

1. "We propose that the metric is not factorizable, but rather the four-dimensional metric is multiplied by a 'warp' factor which is a rapidly changing function of an additional dimension." (Sec. 1, PDF p. 2)
2. "To generate a large hierarchy does not require extremely large $r_c$. This is because the source of the hierarchy is an exponential function of the compactification radius." (Sec. 1, PDF p. 2)
3. The hierarchy between $M$ and $\mu_c = 1/r_c$ "is only of order 50, as opposed to $(M_{Pl}/\mathrm{TeV})^{2/n}$"; "There is one additional dimension, as opposed to $n \ge 2$." (Sec. 1, PDF p. 2)
4. "There are no light Kaluza-Klein modes. The excitation scale is of order a TeV." and "The coupling of an individual KK excitation to matter or to other gravitational modes is set by the weak, not the Planck scale." (Sec. 1, PDF pp. 2-3)
5. The set-up "requires nothing beyond the existence of the 3-branes in five dimensions and their compatibility with four-dimensional Poincare invariance." (Sec. 2, PDF p. 3)
6. "These relations between the boundary and bulk cosmological terms are required in order to obtain a solution that respects four-dimensional Poincare invariance." and "precisely these relations arise in the five-dimensional effective theory of the Horava-Witten scenario" with frozen Calabi-Yau moduli (Sec. 3, PDF p. 5)
7. "We will assume that $k < M$ so that the bulk curvature is small compared to the higher dimensional Planck scale and we trust our solution." (Sec. 3, PDF p. 5)
8. In contrast to ordinary KK reduction, "in the present case there are no such isometries in the presence of the 3-branes. So all such off-diagonal fluctuations of the metric [$A_\mu dx^\mu d\phi$] are massive and excluded from the low-energy effective theory." (Sec. 4, PDF pp. 5-6)
9. "The masses and couplings of the Kaluza-Klein modes are determined by the TeV scale." (Sec. 4, PDF p. 7)
10. "In the 3-brane scenario, where extra-dimensional translational symmetry is necessarily broken, non-trivial warp factors naturally arise upon solving Einstein's equations. The Kaluza-Klein reduction is considerably more subtle than in product spacetimes." (Sec. 5, PDF p. 8)

# Caveats

- Radius stabilisation: $r_c$ is an arbitrary integration constant of the classical solution (PDF p. 5); it is the vev of the modulus $T(x)$ of eq (13). "It will be critical that the $T$ modulus is stabilized at its vacuum expectation value $r_c$, with a mass of at least $10^{-4}$ eV. Although an essential element of the theory, this problem is not yet solved (but see Refs. [8])" (PDF p. 5), and in the Conclusions: "it is important to the viability of our mechanism that it is possible to stabilize the compactification radius roughly two orders of magnitude larger than the fundamental five dimensional Planck length" (PDF p. 8). The paper offers no stabilisation mechanism (the Goldberger-Wise mechanism is later work).
- Fine tuning of tensions: eq (11) fixes both brane tensions in terms of $\Lambda$ and $M$; this is required by 4D Poincare invariance and is not explained dynamically. The visible brane has negative tension, $V_{vis} = -24M^3k$. The paper does not itself say "fine tuning" here (that wording appears in hep-th/9906064); it says the relations "are required" and points to the Horava-Witten analogue.
- Jump conditions: the brane delta functions are handled by treating $\sigma$ as a periodic even function on $-\pi \le \phi \le \pi$ so that $\sigma'' $ from eq (9) contains $\delta(\phi) - \delta(\phi-\pi)$, eq (10); the orbifold identification $\phi \to -\phi$ is what forces $|\phi|$ in eq (9). No general Israel junction formalism is used; the matching is read off from eqs (8) and (10). The sign of $\delta(\phi-\pi)$ follows from the kink of $|\phi|$ at $\phi = \pm\pi$ being opposite to that at $\phi=0$ when the function is continued periodically.
- Planck-mass normalisation: eq (16) follows from eq (15) only if the 4D action is matched as $2M_{Pl}^2\sqrt{-\bar g}\bar R$ (same convention as $2M^3R$ in 5D): $2M^3 r_c\int d\phi\,e^{-2kr_c|\phi|} = 2M^3(1-e^{-2kr_c\pi})/k$, so $M_{Pl}^2 = (M^3/k)(1-e^{-2kr_c\pi})$. With the standard reduced-Planck-mass convention $(M_{Pl}^2/2)\bar R$ one would instead get $M_{Pl}^2 = 4(M^3/k)(1-e^{-2kr_c\pi})$. The tutorial should state which convention it uses; the exponential dependence is convention-independent.
- Validity: requires $k < M$ (small bulk curvature) and $r_c > 1/k$ (large $kr_c$ so that the exponential dominates and $M_{Pl}$ is $r_c$-independent). In the "TeV-fundamental" viewpoint, the strong-coupling / string scale as seen on the visible brane is of order a TeV (PDF p. 8).
- Equation (2) vs (12): eq (2) in the Introduction omits the absolute value $|\phi|$ which is essential on the orbifold; eq (12) is the correct form.
- The running head misprints the arXiv archive as hep-th; cite as hep-ph/9905221.
- The explicit KK decomposition of the graviton and the TeV-scale couplings are only asserted here and deferred to "Ref. [9]" (in preparation); the single-brane analysis is in hep-th/9906064 (randall1999b).
