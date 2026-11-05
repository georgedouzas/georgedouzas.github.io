# Identity

- arXiv: hep-ph/9803315 (v1, 11 Mar 1998). Preprint numbers SLAC-PUB-7769, SU-ITP-98/13.
- Title: "The Hierarchy Problem and New Dimensions at a Millimeter".
- Authors: Nima Arkani-Hamed (SLAC), Savas Dimopoulos (Stanford), Gia Dvali (ICTP Trieste).
- Year: 1998. Venue: not stated in the PDF (preprint; the published version is Phys. Lett. B429 (1998) 263, not stated in the file).
- PDF is 17 pages; printed page p = PDF page p+1 (title/abstract page is unnumbered).

# Structure

1. Abstract — PDF p. 1 (unnumbered).
2. Section 1, Introduction — PDF pp. 2-6 (printed 1-5). Contains the Gauss-law argument, eqs (1)-(4), the localization requirement, and the "third" signature (escape into the bulk, p_T cutoff, "fireworks").
3. Section 2, Phenomenological and Astrophysical Constraints — PDF pp. 6-8 (printed 5-7). KK tower, eqs (5)-(8), lab bounds (Upsilon, Z decays), astrophysical bounds (sun, red giants, SN 1987A), cosmic rays.
4. Section 3, Construction of a Realistic Model — PDF pp. 8-15 (printed 7-14). Vortex in 6D, eqs (9)-(10).
   - 3.1 Localization of Fermions — PDF pp. 10-11 (printed 9-10), eqs (11)-(16).
   - 3.2 Localization of Higgs Scalars — PDF pp. 11-12 (printed 10-11), eqs (17)-(19).
   - 3.3 Localization of Gauge Fields — PDF pp. 12-14 (printed 11-13), eqs (20)-(22).
   - 3.4 A Realistic Theory — PDF pp. 14-15 (printed 13-14), eqs (23)-(25); Pati-Salam SU(4) x SU(2)_R x SU(2)_L in the bulk.
5. Section 4, Summary and Outlook — PDF pp. 15-16 (printed 14-15). Acknowledgments at end of PDF p. 16.
6. References [1]-[12] — PDF p. 17 (printed 16).

# Conventions

- Natural units throughout (hbar = c = 1); lengths quoted in cm, mm, micrometres; energies in GeV/TeV.
- 4D Planck mass: $M_{Pl} = G_N^{-1/2} \sim 10^{18}$ GeV (Sec. 1, PDF p. 2; abstract writes $M_{Pl} \sim G_N^{-1/2}$). Note the paper writes $10^{18}$ GeV, i.e. an order-of-magnitude value, not $1.2\times10^{19}$ GeV; no distinction between reduced and non-reduced Planck mass is made.
- Electroweak scale $m_{EW} \sim 10^3$ GeV; the hierarchy is $m_{EW}/M_{Pl} \sim 10^{-17}$ (PDF p. 2).
- Fundamental scale: $M_{Pl(4+n)}$, "the Planck scale of this $(4+n)$ dimensional theory", taken to be $\sim m_{EW}$ (PDF p. 3). It is defined operationally through the $(4+n)$-dimensional Gauss law, eq (1): the gravitational potential between test masses is $\propto 1/M_{Pl(4+n)}^{n+2}$. No explicit $(4+n)$-dimensional Einstein-Hilbert action or normalisation (no factors of $2\pi$, $8\pi$, volumes of spheres) is written anywhere; all relations are order-of-magnitude ("$\sim$").
- Geometry: spacetime is $R^4 \times M_n$, $n \ge 2$, $M_n$ an $n$-dimensional compact manifold of volume $R^n$ (PDF p. 4). In Sec. 3 the 6D metric signature is $g_{AB} = (-1,1,1,1,1,1)$ (PDF p. 8).
- Bulk graviton coupling (PDF p. 7): $g_{AB} = \eta_{AB} + h_{AB}/\sqrt{m_{EW}^{2+n}}$ with $h_{AB}$ the canonically normalised graviton in $4+n$ dimensions. In Sec. 3 the UV cutoff / vortex inverse thickness is called $\Lambda \sim m_{EW}$.

# Equations the tutorial uses

(a) Gauss-law derivation (Sec. 1, PDF p. 3, printed p. 2). Two test masses $m_1, m_2$ at separation $r \ll R$ feel the $(4+n)$-dimensional potential
$$V(r) \sim \frac{m_1 m_2}{M_{Pl(4+n)}^{n+2}}\,\frac{1}{r^{n+1}}, \quad (r \ll R) \qquad \text{(1)}$$
For $r \gg R$ "their gravitational flux lines can not continue to penetrate in the extra dimensions, and the usual $1/r$ potential is obtained"
$$V(r) \sim \frac{m_1 m_2}{M_{Pl(4+n)}^{n+2} R^n}\,\frac{1}{r}, \quad (r \gg R) \qquad \text{(2)}$$
"so our effective 4 dimensional $M_{Pl}$ is"
$$M_{Pl}^2 \sim M_{Pl(4+n)}^{2+n} R^n. \qquad \text{(3)}$$
Exactly as printed: no $2\pi$ or angular factors; $R^n$ is the volume of $M_n$ (PDF p. 4).

(b) Numerical estimate (Sec. 1, PDF p. 3, eq 4). "Putting $M_{Pl(4+n)} \sim m_{EW}$ and demanding that $R$ be chosen to reproduce the observed $M_{Pl}$ yields"
$$R \sim 10^{\frac{30}{n}-17}\,\text{cm} \times \left(\frac{1\,\text{TeV}}{m_{EW}}\right)^{1+\frac{2}{n}}. \qquad \text{(4)}$$
Printed consequences: $n=1$: $R \sim 10^{13}$ cm ("deviations from Newtonian gravity over solar system distances, so this case is empirically excluded"); $n = 2$: $R \sim 100\,\mu\text{m} - 1$ mm (PDF p. 3). Sec. 3 uses $R \sim 1$ mm for $n=2$ (PDF p. 8). The paper gives no table for $n \ge 3$; from eq (4) one gets $n=3$: $\sim 10^{-7}$ cm, $n=6$: $\sim 10^{-12}$ cm, etc. (tutorial's own evaluation, not printed). Sec. 4 (PDF p. 16): "if there are many new dimensions, their size -- given by eq (4) -- approaches the weak scale and there is no large hierarchy". Sec. 2 (PDF p. 8): for $n \gtrsim 7$, $1/R \gtrsim 100$ MeV.

(c) Why SM fields are confined (Sec. 1, PDF p. 3, and summary PDF p. 4). Printed argument: "While gravity has not been probed at distances smaller than a millimeter, the SM gauge forces have certainly been accurately measured at weak scale distances. Therefore, the SM particles cannot freely propagate in the extra $n$ dimension, but must be localized to a 4 dimensional submanifold. Since we assume that $m_{EW}$ is the only short-distance scale in the theory, our 4-dimensional world should have a 'thickness' $\sim m_{EW}^{-1}$ in the extra $n$ dimensions. The only fields propagating in the $(4+n)$ dimensional bulk are the $(4+n)$ dimensional graviton, with couplings suppressed by the $(4+n)$ dimensional Planck mass $\sim m_{EW}$." The word "brane" is not used; the paper says "4-dimensional submanifold", "4-dimensional manifold of weak scale thickness", "our Universe", "vortex", "wall". Localization is explicitly the "non-trivial task" (PDF p. 4) and Sec. 3 offers a vortex construction as an "existence proof" (PDF pp. 4, 8), with D-brane constructions mentioned as an alternative (footnote, PDF p. 12; PDF p. 14).

(d) KK graviton tower (Sec. 2, PDF p. 6, printed p. 5). Stated in words, no equation number: "in four dimensional language -- our theory consists of the Standard model together with the graviton and all its Kaluza-Klein (KK) excitations recurring once every $1/R$, per extra dimension $n$." All are called "gravitons" independent of mass; each couples with strength $\sim 1/M_{Pl}$; the multiplicity beneath energy $E$ is $(ER)^n$. Total rate for emitting any one of the available gravitons:
$$\sim \frac{1}{M_{Pl}^2}(\Delta E R)^n \qquad \text{(5)}$$
"Using eq (3) we can rewrite this as"
$$\sim \frac{\Delta E^n}{m_{EW}^{2+n}} \qquad \text{(6)}$$
Branching ratio for emitting a graviton in any process:
$$\sim (\Delta E/m_{EW})^{2+n} \qquad \text{(7)}$$
Dictionary to goldstone-boson bounds ($F$ = goldstone decay constant):
$$1/F^2 \longleftrightarrow \Delta E^n / m_{EW}^{2+n} \qquad \text{(8)}$$
Collider signatures summarised (PDF pp. 2, 5-7, 15): (i) strong quantum-gravitational interactions at LHC/NLC at $m_{EW} \sim$ TeV; (ii) particles kicked off into the bulk above an escape energy $E_{esc} \gtrsim m_{EW}$, giving "a sharp upper limit to the transverse momentum" at $p_T = E_{esc}$ and apparent energy non-conservation (charge cannot be lost; a flux tube of tension $\sim m_{EW}^2$ pulls charged particles back, PDF pp. 5-6); (iii) for certain compact manifolds, escaped particles return with frequency $R^{-1} \sim 10^{27-30/n}$ Hz ("fireworks", mm displaced vertices for $n=2$, PDF p. 6); (iv) graviton emission looks like missing-energy searches but carried by massless particles (PDF p. 7). Astrophysical: gravitons behave like goldstones/axions/neutrinos in cooling stars (sun, red giants, SN 1987A), PDF p. 7.

(e) Bounds (Sec. 1 PDF p. 3; Sec. 2 PDF pp. 7-8). $n = 1$ excluded (solar-system deviations, $R \sim 10^{13}$ cm). $n = 2$: $1/r^2 \to 1/r^4$ transition at $\sim 100\,\mu$m-1 mm, testable by planned sub-mm gravity experiments [11]. Lab: BR($\Upsilon \to$ graviton) $\sim 10^{-8}$ unobservable; BR($Z \to X +$ graviton) up to $\sim 10^{-5}$, "the strongest laboratory constraints". Sun: effective $F \sim 10^{12}$ GeV vs probed $\sim 10^7$ GeV, safe. Red giants: $F \sim 10^{10}$ GeV, one order above the limit. SN 1987A ($\Delta E$ = 20 MeV, $m_{EW}$ = 1 TeV, $n = 2$): effective $F \sim 10^8$ GeV, below the claimed lower limit $10^{10}$ GeV, so an "interesting constraint"; "easily satisfied if $n > 2$ or if $m_{EW} > 10$ TeV"; for $n \gtrsim 7$ ($1/R \gtrsim 100$ MeV) no astrophysical bound applies. Cosmic rays: no significant constraint (soft QCD interactions). Nucleosynthesis unaffected (PDF p. 16).

# Attributable statements

1. The Planck scale "is not a fundamental scale; its enormity is simply a consequence of the large size of the new dimensions" (Abstract, PDF p. 1; Sec. 1, PDF p. 4).
2. Gravity "has only been accurately measured in the $\sim 1$ cm range"; assuming $M_{Pl}$ fundamental extrapolates Newtonian gravity over "33 orders of magnitude" (Sec. 1, PDF p. 2).
3. The weakness of 4D gravity follows from Gauss's law in $4+n$ dimensions: $M_{Pl}^2 \sim M_{Pl(4+n)}^{2+n} R^n$, eqs (1)-(3) (Sec. 1, PDF p. 3).
4. With $M_{Pl(4+n)} \sim m_{EW}$, $R \sim 10^{30/n-17}$ cm; $n=1$ is excluded, $n=2$ gives $R \sim 100\,\mu$m-1 mm (eq 4, Sec. 1, PDF p. 3).
5. SM fields must be localized to a 4D submanifold of thickness $\sim m_{EW}^{-1}$ because gauge forces are measured at weak-scale distances; only the graviton lives in the bulk (Sec. 1, PDF pp. 3-4).
6. In 4D language the bulk graviton is a KK tower "recurring once every $1/R$ per extra dimension", with multiplicity $(ER)^n$ below energy $E$; the combined emission rate is $\sim \Delta E^n/m_{EW}^{2+n}$ (Sec. 2, PDF p. 6, eqs 5-6).
7. SN 1987A is the most constraining astrophysical bound for $n=2$, $m_{EW} = 1$ TeV; satisfied for $n > 2$ or $m_{EW} > 10$ TeV (Sec. 2, PDF pp. 7-8).
8. The mm scale "is not a Lagrangean parameter that needs to be stabilized by a symmetry... It is a parameter characterizing a solution, the size of the two extra dimensions"; why it is mm-sized is "not addressed" (Sec. 4, PDF p. 16).
9. The vortex model of Sec. 3 "must be viewed at best as an 'existence proof'"; D-branes offer another localization route (Sec. 1, PDF p. 4; Sec. 3, PDF pp. 8, 14).
10. Supersymmetry is no longer needed for the hierarchy, but "may still be crucial" for the quantum gravity theory above $m_{EW}$ (Sec. 1, PDF p. 4).

# Caveats

- All relations in Secs. 1-2 are order-of-magnitude ("$\sim$"); eq (3) omits the $2\pi$/sphere-volume factors a tutorial would obtain from the actual KK reduction ($M_{Pl}^2 = M_*^{n+2} V_n$ with $V_n = (2\pi R)^n$ for a torus). The paper's $R$ is defined by "volume $R^n$" (PDF p. 4), so its $R$ is a volume$^{1/n}$, not a circle radius.
- $M_{Pl} \sim 10^{18}$ GeV is quoted loosely; $G_N^{-1/2} = 1.22\times10^{19}$ GeV. Eq (4)'s $10^{30/n-17}$ cm should be read as a logarithmic estimate.
- No formula $m_k = k/R$ for KK masses is written; the tower is described verbally (PDF p. 6). Likewise no equation for the $1/r^2 \to 1/r^4$ crossover beyond eqs (1)-(2).
- The footnote on PDF p. 5 distinguishes this scenario from standard KK: here particles with momentum in the extra dimensions have TeV energies, wavelengths $\ll R$, and "simply escape", rather than appearing as 4D KK particles of mass $1/R$.
- Localization of SM fields is assumed, not derived; the explicit vortex model (Sec. 3) relies on a postulated bulk mass gap $\sim \Lambda$ via a strongly coupled Pati-Salam group (conditions 1-3, PDF p. 14), and a sphere vortex-anti-vortex pair is only metastable (PDF p. 9).
- Dangerous higher-dimension operators (proton decay, $K$-$\bar K$ mixing) suppressed only by $m_{EW}$ are "simply assumed" absent (PDF pp. 4, 7).
- The SN 1987A bound is stated as marginally violated for $n=2$, $m_{EW}=1$ TeV (effective $F \sim 10^8$ vs limit $10^{10}$ GeV), i.e. the paper's own numbers already push $n=2$ to $m_{EW} \gtrsim 10$ TeV; later literature sharpened these bounds.
- Venue is not printed in the PDF; the arXiv stamp reads "11 Mar 1998" and the compile date line reads "November 26, 2024" (a re-rendering artefact, not the paper's date).

# Sections and equations used (part-ii-08-renormalisation)

- Sec. 1, eqs. 1-3 (Gauss's law in 4 + n dimensions, M_Pl^2 ~ M^(2+n) R^n, sizes of the extra dimensions): exm-ren-sc-gravity,
  exr-ren-sc-add.
