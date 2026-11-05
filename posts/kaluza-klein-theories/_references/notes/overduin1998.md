# Identity

- arXiv: gr-qc/9805018 (v1, 7 May 1998)
- Title: Kaluza-Klein Gravity
- Authors: J. M. Overduin (Univ. of Victoria) and P. S. Wesson (Univ. of Waterloo; Gravity Probe-B, Stanford)
- Year: 1998 (arXiv); the PDF footer says only "Preprint submitted to Elsevier Preprint". No journal is stated in the PDF. (Background knowledge, not from the PDF: the published version is Phys. Rep. 283 (1997) 303.)
- Length: 102 PDF pages; printed page numbers coincide with PDF page numbers (title page = p. 1, references pp. 81-102).
- Scope statement (p. 2, Sec. 1): a review "from the general relativity, rather than particle physics side"; three approaches are compared: compactified, projective, noncompactified. The compactified-theory sections are deliberately kept short because the subject "has been thoroughly reviewed elsewhere" (p. 20, refs [3-7], [8-14], [195]).

# Structure

PDF page = printed page throughout.

1. Introduction, p. 2
2. Historical Overview, p. 3
   - 2.1 Higher Dimensions, p. 3
   - 2.2 Kaluza-Klein Theory, p. 3
   - 2.3 Approaches to Higher-Dimensional Unification, p. 5
   - 2.4 The Compactified Approach, p. 6
   - 2.5 Compactification Mechanisms, p. 7
   - 2.6 D = 11 Supergravity, p. 8
   - 2.7 D = 10 Superstring Theory, p. 10
   - 2.8 The Projective Approach, p. 11
   - 2.9 The Noncompactified Approach, p. 11
3. The Kaluza Mechanism, p. 12  [core for the tutorial]
   - 3.1 Matter from Geometry, p. 13
   - 3.2 A Minimal Extension of General Relativity, p. 13
   - 3.3 The Cylinder Condition, p. 14
   - 3.4 The Case phi = constant, p. 15
   - 3.5 The Case A_alpha = 0: Brans-Dicke Theory, p. 16
   - 3.6 Conformal Rescaling, p. 17
   - 3.7 Conformal Ambiguity, p. 18
4. Compactified Theories, p. 20  [core for the tutorial]
   - 4.1 Klein's Compactification Mechanism, p. 20
   - 4.2 Quantization of Charge, p. 21
   - 4.3 Extension to Higher Dimensions, p. 23
   - 4.4 Higher-Dimensional Matter, p. 25
5. Projective Theories, p. 27 (5.1 p. 27, 5.2 p. 27)  [not needed]
6. Noncompactified Theories, p. 29 (6.1 The Metric p. 30; 6.2 The Field Equations p. 31; 6.3 p. 32; 6.4 p. 34; 6.5 p. 37; 6.6 p. 39; 6.7 p. 42; 6.8 p. 44; 6.9 p. 46; 6.10 p. 48)  [induced-matter approach; not needed]
7. Cosmology, p. 49 (7.1 p. 49; 7.2 p. 51; 7.3 p. 53; 7.4 p. 55; 7.5 p. 56; 7.6 p. 58)  [not needed]
8. Astrophysics, p. 60 (8.1 p. 60; 8.2 p. 62; 8.3 p. 64; 8.4 p. 67; 8.5 p. 68; 8.6 p. 70; 8.7 p. 70; 8.8 p. 72; 8.9 p. 73; 8.10 p. 74; 8.11 p. 77)  [not needed]
9. Conclusions, p. 80
- Acknowledgements and References, pp. 81-102

# Conventions

- Index conventions (p. 13, p. 14): capital Latin A, B, ... run over 0,1,2,3,4; Greek alpha, beta, ... over 0,1,2,3; small Latin a, b, ... over 1,2,3. Five-dimensional quantities carry hats. The fifth coordinate is y = x^4 (p. 13).
- Metric signature: four-dimensional signature (+ - - -) (p. 14). The fifth dimension is intended to be spacelike: the ground-state 5D metric is diag(eta_{alphabeta}, -1), eq. (21), p. 21. Note, however, that ansatz (5) is printed with g_44 = +phi^2, which is not literally consistent with (21) for real phi; the paper does not comment. In Sec. 6 they introduce epsilon (epsilon^2 = 1) in g_44 = epsilon phi^2 to allow either signature, eq. (47), p. 30, and remark that a timelike fifth dimension flips the sign of the Maxwell term in (9) and of m_n (p. 30), so the spacelike case corresponds to the physical sign choice.
- Ricci tensor and Christoffel symbols, eq. (4), p. 13 (quoted exactly, "defined in terms of the metric exactly as in four dimensions"):
  \hat R_{AB} = \partial_C \hat\Gamma^C_{AB} - \partial_B \hat\Gamma^C_{AC} + \hat\Gamma^C_{AB}\hat\Gamma^D_{CD} - \hat\Gamma^C_{AD}\hat\Gamma^D_{BC},
  \hat\Gamma^C_{AB} = \tfrac12 \hat g^{CD}(\partial_A \hat g_{DB} + \partial_B \hat g_{DA} - \partial_D \hat g_{AB}).
  Ricci scalar: \hat R = \hat g_{AB} \hat R^{AB} (p. 13). Einstein tensor: \hat G_{AB} \equiv \hat R_{AB} - \hat R \hat g_{AB}/2 (p. 13); G_{alphabeta} \equiv R_{alphabeta} - R g_{alphabeta}/2 (p. 14).
  The Riemann tensor is NOT written anywhere in the paper (not found). The Ricci formula above is the contraction R_{BD} = R^C{}_{BCD} of R^A{}_{BCD} = \partial_C \Gamma^A_{BD} - \partial_D \Gamma^A_{BC} + \Gamma^A_{CE}\Gamma^E_{BD} - \Gamma^A_{DE}\Gamma^E_{BC} (Landau-Lifshitz/MTW sign; this is my inference, not a statement in the paper).
- Action sign: S = -(1/16 pi \hat G) \int \hat R \sqrt{-\hat g} d^4x dy, eq. (3), p. 13. Together with (+ - - -) this is the Landau-Lifshitz convention.
- 5D metric ansatz, eq. (5), p. 14 (exact form):
  (\hat g_{AB}) = \begin{pmatrix} g_{\alpha\beta} + \kappa^2\phi^2 A_\alpha A_\beta & \kappa^2\phi^2 A_\alpha \\ \kappa^2\phi^2 A_\beta & \phi^2 \end{pmatrix}.
  The constant kappa scales A_alpha "in order to get the right multiplicative factors in the action later on" (p. 14). No exponential parametrisation is used; the scalar enters as phi^2 in g_44 and there is no overall conformal factor in (5). The conformally rescaled version (16), p. 18, has an overall phi^{-1/3} and uses phi instead of phi^2 (after the redefinition phi^2 -> phi).
- Field strength: F_{\alpha\beta} \equiv \partial_\alpha A_\beta - \partial_\beta A_\alpha (p. 14). EM energy-momentum tensor: T^{EM}_{\alpha\beta} \equiv g_{\alpha\beta} F_{\gamma\delta}F^{\gamma\delta}/4 - F_\alpha{}^\gamma F_{\beta\gamma} (p. 14).
- Coupling/normalisation: kappa \equiv 4\sqrt{\pi G}, eq. (8), p. 15 (i.e. kappa^2 = 16 pi G; NOT the common kappa^2 = 8 pi G). The 4D Newton constant is G \equiv \hat G / \int dy, eq. (10), p. 15. Minimal coupling in QED written as \partial_\alpha -> \partial_\alpha + i e A_\alpha (p. 22); fine-structure constant alpha \equiv q_1^2/4\pi (eq. (26), p. 22), i.e. Heaviside-Lorentz-type rationalised units with hbar = c = 1.
- Units (p. 14): c = 1 throughout; hbar = 1 in Sec. 3 (and implicitly in Sec. 4, where m_n = |n|/(r sqrt(phi)) and l_pl = sqrt(G)); G = 1 in Secs. 7 and 8. Planck length l_pl = \sqrt{G} (p. 22), m_pl ~ 10^19 GeV (p. 21).

# Equations the tutorial uses

## (a) 5D line element ansatz

The paper never writes ds^2 explicitly; it gives the metric components, eq. (5), p. 14:
  (\hat g_{AB}) = \begin{pmatrix} g_{\alpha\beta} + \kappa^2\phi^2 A_\alpha A_\beta & \kappa^2\phi^2 A_\alpha \\ \kappa^2\phi^2 A_\beta & \phi^2 \end{pmatrix}, \quad y = x^4.
Equivalent line element (my rewriting, not in the paper): d\hat s^2 = g_{\alpha\beta}dx^\alpha dx^\beta + \phi^2 (dy + \kappa A_\alpha dx^\alpha)^2.
Special cases quoted by the paper:
- A_alpha = 0 (Brans-Dicke sector), eq. (11), p. 16: (\hat g_{AB}) = diag(g_{\alpha\beta}, \phi^2).
- Ground state ("vacuum expectation value of the full metric"), eq. (21), p. 21: (\langle \hat g_{AB}\rangle) = diag(\eta_{\alpha\beta}, -1), topology M^4 x S^1.
- Conformally rescaled ansatz (after phi^2 -> phi and Omega^2 = phi^{-1/3}), eq. (16), p. 18:
  (\hat g'_{AB}) = \phi^{-1/3}\begin{pmatrix} g_{\alpha\beta} + \kappa^2\phi A_\alpha A_\beta & \kappa\phi A_\alpha \\ \kappa\phi A_\beta & \phi \end{pmatrix}.
- Higher-dimensional (4+d) KK ansatz (n = 0 modes), eq. (31), p. 24:
  (\hat g^{(0)}_{AB}) = \begin{pmatrix} g_{\alpha\beta} + \tilde g_{\mu\nu} K^\mu_i A^i_\alpha K^\nu_j A^j_\beta & \tilde g_{\mu\nu} K^\mu_i A^i_\alpha \\ \tilde g_{\mu\nu} K^\nu_i A^i_\beta & \tilde g_{\mu\nu} \end{pmatrix},
  with \tilde g_{\mu\nu} the metric of the d-dimensional compact space (mu, nu = 1..d), K^\mu_i (i = 1..n) linearly independent Killing vectors satisfying eq. (33), p. 24:
  \frac{\partial K^\lambda_i}{\partial y^\mu}\tilde g_{\lambda\nu} + \frac{\partial K^\lambda_i}{\partial y^\nu}\tilde g_{\lambda\mu} + K^\lambda_i \frac{\partial \tilde g_{\mu\nu}}{\partial y^\lambda} = 0.
  (Note: no coupling constant kappa and no scalar appear in (31) as printed.)

## (b) 5D action and field equations

- Vacuum 5D Einstein equations, eqs. (1), (2), p. 13: \hat G_{AB} = 0, equivalently \hat R_{AB} = 0.
- 5D Einstein-Hilbert action, eq. (3), p. 13:
  S = -\frac{1}{16\pi\hat G}\int \hat R \sqrt{-\hat g}\, d^4x\, dy,
  with \hat G the "five-dimensional gravitational constant".

## (c) Reduced 4D equations and action

- Reduced field equations (cylinder condition applied to (5) with (4)), eq. (6), p. 14; the alpha-beta, alpha-4 and 44 components of (2) become respectively:
  G_{\alpha\beta} = \frac{\kappa^2\phi^2}{2} T^{EM}_{\alpha\beta} - \frac{1}{\phi}\left[\nabla_\alpha(\partial_\beta\phi) - g_{\alpha\beta}\Box\phi\right],
  \nabla^\alpha F_{\alpha\beta} = -3\frac{\partial^\alpha\phi}{\phi}F_{\alpha\beta},
  \Box\phi = \frac{\kappa^2\phi^3}{4}F_{\alpha\beta}F^{\alpha\beta}.
  Count: 10 + 4 + 1 = 15 equations, matching the 15 independent components of (5) (p. 14). Attributed to refs [15, 41] (Lessner; Thiry).
- phi = const, eq. (7), p. 15: G_{\alpha\beta} = 8\pi G\phi^2 T^{EM}_{\alpha\beta}, \nabla^\alpha F_{\alpha\beta} = 0, with the identification kappa \equiv 4\sqrt{\pi G}, eq. (8), p. 15. Kaluza and Klein set phi = 1 (p. 15).
- Reduced 4D action (Jordan frame; cylinder condition used to drop y-derivatives and to pull \int dy out), eq. (9), p. 15, attributed to ref [10] (Appelquist):
  S = -\int d^4x\,\sqrt{-g}\,\phi\left(\frac{R}{16\pi G} + \frac14\phi^2 F_{\alpha\beta}F^{\alpha\beta} + \frac{2}{3\kappa^2}\frac{\partial^\alpha\phi\,\partial_\alpha\phi}{\phi^2}\right),
  with G \equiv \hat G/\int dy, eq. (10), p. 15, and eq. (8) used "to bring the factor of 16 pi G inside the integral". So: F^2 coefficient = -(1/4) phi^3 (overall), scalar kinetic coefficient = -(2/3 kappa^2) (\partial\phi)^2/\phi (overall). For phi = const the first two terms are the Einstein-Maxwell action "scaled by factors of phi"; the third is a massless Klein-Gordon action (p. 15).
- A_alpha = 0 case, eq. (12), p. 16: S = -\frac{1}{16\pi G}\int d^4x\sqrt{-g}\,R\phi, which is the omega = 0 case of the Brans-Dicke action, eq. (13), p. 16:
  S_{BD} = -\int d^4x\sqrt{-g}\left(\frac{R\phi}{16\pi G} + \omega\frac{\partial^\alpha\phi\,\partial_\alpha\phi}{\phi}\right) + S_m.
  Observation requires omega > ~500 [197], so "this simple model is certainly not viable" (p. 17).
- Conformal (Weyl) rescaling to canonical (Einstein) form, Sec. 3.6, pp. 17-18 (the paper does not use the words "Einstein frame"/"Jordan frame"; it says "Pauli metric" vs "Jordan metric", p. 18):
  eq. (14), p. 17: \hat g_{AB} \to \hat g'_{AB} = \Omega^2 \hat g_{AB}, Omega^2 > 0 a function of x^0..x^3 only;
  eq. (15), p. 17 (4D Ricci scalar under g_{\alpha\beta} \to \Omega^2 g_{\alpha\beta}, ref [203]): R \to R' = \Omega^{-2}\left(R + 6\frac{\Box\Omega}{\Omega}\right);
  choice: redefine phi^2 -> phi, then Omega^2 = phi^{-1/3} (p. 17), giving metric (16) and the rescaled action, eq. (17), p. 18 (refs [8,12,13]):
  S' = -\int d^4x\sqrt{-g'}\left(\frac{R'}{16\pi G} + \frac14\phi F'_{\alpha\beta}F'^{\alpha\beta} + \frac{1}{6\kappa^2}\frac{\partial'^\alpha\phi\,\partial'_\alpha\phi}{\phi^2}\right),
  primes = quantities built from the rescaled metric (\partial'^\alpha\phi = g'^{\alpha\beta}\partial_\beta\phi); G, kappa as before.
  A_alpha = 0 rescaled, eq. (18), p. 18: S' = -\int d^4x\sqrt{-g'}\left(\frac{R'}{16\pi G} + \frac{1}{6\kappa^2}\frac{\partial'^\alpha\phi\,\partial'_\alpha\phi}{\phi^2}\right);
  dilaton sigma \equiv \ln\phi/(\sqrt3\,\kappa), eq. (19), p. 18: S' = -\int d^4x\sqrt{-g'}\left(\frac{R'}{16\pi G} + \frac12\partial'^\alpha\sigma\,\partial'_\alpha\sigma\right), "the canonical action for a minimally coupled scalar field with no potential" [204].

## (d) Transformation of A_alpha under a shift of the fifth coordinate

Sec. 4.3, p. 23. Assuming circular topology and small scale, the theory is invariant under
  eq. (28): y \to y' = y + f(x), x = (x^0, x^1, x^2, x^3);
  eq. (29) (5D tensor law): \hat g_{AB} \to \hat g'_{AB} = \frac{\partial x^C}{\partial x'^A}\frac{\partial x^D}{\partial x'^B}\hat g_{CD};
  eq. (30): "the only change to the metric (5) is" A_\alpha \to A'_\alpha = A_\alpha + \partial_\alpha f(x),
"which is just a U(1) local gauge transformation" (p. 24). (Note: as printed, no factor of 1/kappa appears in (30), although A_alpha enters (5) multiplied by kappa; see Caveats.)
Higher-dimensional analogue, eqs. (32), (34), p. 24: y^\mu \to y'^\mu = y^\mu + \sum_{i=1}^n \varepsilon^i(x) K^\mu_i, and A^i_\alpha \to A^{i\prime}_\alpha = A^i_\alpha + \partial_\alpha\varepsilon^i(x), gauge group = isometry group of the compact manifold (p. 24).

## (e) Mode expansion, masses, charges, radius-Planck-coupling relation

- Periodicity (p. 20): f(x, y) = f(x, y + 2\pi r), r = "scale parameter or 'radius' of the fifth dimension".
- Fourier expansion, eq. (20), p. 20:
  g_{\alpha\beta}(x,y) = \sum_{n=-\infty}^{\infty} g^{(n)}_{\alpha\beta}(x)e^{iny/r}, \quad A_\alpha(x,y) = \sum_n A^{(n)}_\alpha(x)e^{iny/r}, \quad \phi(x,y) = \sum_n \phi^{(n)}(x)e^{iny/r}.
  Modes carry y-momentum of order |n|/r; the "Kaluza-Klein ansatz" = discard all n != 0 modes (p. 21).
- Massless 5D scalar, eq. (22), p. 21: S_{\hat\psi} = -\int d^4x\,dy\sqrt{-\hat g}\,\partial^A\hat\psi\,\partial_A\hat\psi; expansion eq. (23), p. 22: \hat\psi(x,y) = \sum_n \hat\psi^{(n)}e^{iny/r}.
- Reduced scalar action (using the rescaled metric (16)), eq. (24), p. 22, ref [8]:
  S_{\hat\psi} = -\left(\int dy\right)\sum_n\int d^4x\sqrt{-g}\left[\left(\partial^\alpha + \frac{in\kappa A^\alpha}{r}\right)\hat\psi^{(n)}\left(\partial_\alpha + \frac{in\kappa A_\alpha}{r}\right)\hat\psi^{(n)} - \frac{n^2}{\phi\, r^2}\hat\psi^{(n)2}\right].
  (Transcribed as printed: no complex conjugate on either \hat\psi^{(n)}.)
- Charge of the n-th mode, eq. (25), p. 22 (compare with \partial_\alpha \to \partial_\alpha + ieA_\alpha; A_alpha renormalised by dividing out (\phi\int dy)^{1/2}; uses (8) and (10)):
  q_n = \frac{n\kappa}{r}\left(\phi\int dy\right)^{-1/2} = \frac{n\sqrt{16\pi G}}{r\sqrt\phi}.
- Fine-structure estimate, eq. (26), p. 22, taking r\sqrt\phi ~ l_pl = \sqrt G:
  \alpha \equiv \frac{q_1^2}{4\pi} \sim \frac{(\sqrt{16\pi G}/\sqrt G)^2}{4\pi} = 4.
- Mass of the n-th mode ("square root of the coefficient of the \hat\psi^{(n)2}-term"), eq. (27), p. 22:
  m_n = \frac{|n|}{r\sqrt\phi}.
  With r\sqrt\phi ~ l_pl the n = 1 mass is l_pl^{-1} ~ m_pl ~ 10^19 GeV instead of 0.5 MeV: "some twenty-two orders of magnitude" (pp. 22-23).
- Radius / Planck mass / coupling relation: not stated as a separate formula; it is eq. (25) with n = 1 and hbar = c = 1, G = m_pl^{-2}: q_1 = \sqrt{16\pi}/(m_{pl}\, r\sqrt\phi), equivalently \alpha = 4/(m_{pl}^2 r^2 \phi) (my rewriting of (25)-(26)). Experimental bound: r < 1 am = 10^{-18} m [55]; theorists often set r = l_pl ~ 10^{-35} m (p. 21, p. 6).

## (f) Explicit intermediate quantities

- Inverse metric \hat g^{AB} of ansatz (5): not found.
- Christoffel symbols of ansatz (5): not found (only the general definition (4)).
- Ricci components of ansatz (5) with the cylinder condition: not found; only the final reduced equations (6) are given. The only explicit Ricci components in the paper are eq. (48), p. 31, for the diagonal metric (47) (\hat g_{AB}) = diag(g_{\alpha\beta}, \varepsilon\phi^2) WITHOUT the cylinder condition (noncompactified theory, A_alpha = 0, ref [20]); the cylindrical limit of its first line is \hat R_{\alpha\beta} = R_{\alpha\beta} - \nabla_\beta(\partial_\alpha\phi)/\phi and of its third line \hat R_{44} = -\varepsilon\phi\Box\phi, consistent with the A_alpha = 0 limit of (6).
- Determinant relation \sqrt{-\hat g} = \phi\sqrt{-g} is used implicitly in passing from (3) to (9) but is not written (not found).

# Attributable statements

1. Three key features of Kaluza-type theories: (i) pure geometry, no explicit T_AB; (ii) minimal extension of GR (only the index range changes); (iii) a priori cylindricity. Compactified, projective and noncompactified approaches each sacrifice or modify one of these. (Sec. 2.3, p. 5)
2. Kaluza showed that 5D vacuum GR (^5G_AB = 0) contains 4D GR with an electromagnetic source plus Maxwell's equations; a Klein-Gordon equation for a massless scalar was also present but "suppressed" by Kaluza. (Sec. 2.2, p. 4)
3. Gauge invariance is geometric: U(1) invariance is "added onto" Einstein's equations as invariance under coordinate transformations along the fifth dimension; explicitly y -> y + f(x) induces A_alpha -> A_alpha + d_alpha f, eqs. (28)-(30). (Sec. 2.2, p. 4; Sec. 4.3, pp. 23-24)
4. Cylindricity and compactification are logically distinct; compactification is "only one mechanism by which to explain the apparently four-dimensional nature of the world". Strictly, "Kaluza-Klein theory" should mean cylindricity plus compactification. (Sec. 4, p. 20; Sec. 2.4, p. 7)
5. Klein's mechanism: circular topology plus small scale; all fields are Fourier-expanded (eq. (20)); for small r only the n = 0 modes are observable. (Sec. 4.1, p. 20)
6. phi = const is consistent with the third of the field equations (6) only when F_{alphabeta}F^{alphabeta} = 0, first pointed out by Jordan and Thiry; "the fact that this took twenty years to be acknowledged is a measure of the deep suspicion with which scalar fields were viewed". (Sec. 3.4, p. 15)
7. The reduced action (9) contains three parts: Einstein, Maxwell (scaled by phi) and a massless Klein-Gordon scalar; sourceless 5D equations giving 4D equations with source "constitutes the central miracle of Kaluza-Klein theory". (Sec. 3.4, p. 15)
8. The extra factor phi in (9) means the gravitational action is not canonical unless phi = const; setting phi = 1 by definition is "not a generally covariant procedure"; the fix is the Weyl rescaling Omega^2 = phi^{-1/3} giving (17). (Sec. 3.6, p. 17)
9. The n-th Fourier mode of a 5D scalar carries quantised charge q_n = n sqrt(16 pi G)/(r sqrt(phi)) and mass m_n = |n|/(r sqrt(phi)); with r sqrt(phi) ~ l_pl this gives alpha ~ 4 but m_1 ~ m_pl, a discrepancy of ~22 orders of magnitude, which "played a large role in the abandonment of five-dimensional Kaluza-Klein theory". Modern compactified theories identify light particles with n = 0 modes, give them mass by spontaneous symmetry breaking, and obtain charged massless modes by going to higher dimensions. (Sec. 4.2, pp. 22-23)
10. The higher-dimensional ansatz (31) does not satisfy the (4+d)-dimensional Einstein equations unless the Killing vectors are y-independent, i.e. the compact manifold is flat (the "consistency problem", Duff et al. [11,12]); conventional compactification for d > 1 therefore needs explicit higher-dimensional matter, torsion, or higher-derivative terms. (Sec. 4.4, p. 25; Sec. 2.5, p. 7)
11. Timelike extra dimensions give the wrong sign of the Maxwell action relative to the Einstein one in (9) and the wrong sign of m_n in (24) (tachyons). (Sec. 6.1, p. 30)

# Caveats

- Conventions differ from most particle-physics treatments: signature (+ - - -), action with an overall minus sign, kappa \equiv 4\sqrt{\pi G} (kappa^2 = 16 pi G rather than 8 pi G), and A_alpha scaled by kappa inside the metric (5). Check every factor before comparing with Duff-Nilsson-Pope [12] or Bailin-Love [13], which the paper cites for (17) and (24) without reproducing their conventions.
- Eq. (30) as printed has A_alpha -> A_alpha + d_alpha f with no 1/kappa, although (5) contains kappa A_alpha; the exact coefficient depends on how one normalises f. The paper does not comment on this.
- The role of the scalar: the truncation phi = const (Kaluza/Klein's phi = 1) is inconsistent with the 44 equation unless F^2 = 0 (p. 15). Keeping phi but setting A_alpha = 0 gives omega = 0 Brans-Dicke theory, ruled out by omega > ~500 (pp. 16-17). A_alpha = 0 is a physical restriction (not a coordinate choice) once the cylinder condition is imposed (p. 16).
- Eq. (9) is given in the Jordan-type frame with an overall phi; the "canonical" form (17) requires the rescaling Omega^2 = phi^{-1/3} after the relabelling phi^2 -> phi (pp. 17-18). Keep track of which phi is meant: in (5)-(13) g_44 = phi^2; in (16)-(19) and (24)-(27) the scalar is the redefined phi (= old phi^2).
- Conformal ambiguity (Sec. 3.7, pp. 18-19): which metric is "real" (Jordan vs Pauli) is unresolved; Cho [207, 61] argues only Omega^2 = phi^{-1/3} gives a massless spin-2 graviton and a scalar kinetic term bounded from below; Sokolowski et al. [205, 215, 216] use sqrt(varphi) with varphi = phi^{-3/2}, "the exponent depends on whether one rescales the five-, or only the four-dimensional metric".
- Eq. (24) is stated to use the rescaled metric (16), and A_alpha in (25) is renormalised by dividing out (phi \int dy)^{1/2}; the printed (24) also omits complex conjugates. Treat the numerical coefficients of (24)-(27) as order-of-magnitude for the tutorial and re-derive them.
- The estimate alpha ~ 4 (eq. (26)) assumes r sqrt(phi) = l_pl exactly; the paper itself calls it approximate ("an improved determination of r sqrt(phi) would presumably hit closer to the mark").
- Three distinct metrics in compactified theory (p. 21, after Duff et al. [11,12]): the full Fourier-expanded metric, the KK-ansatz (n = 0) metric, and the ground-state metric <g_AB>; in 5D the ansatz is just dropping y-dependence, but in higher dimensions the relation is "more complicated". Cho [61-64] has suggested abandoning the zero-mode approximation in favour of dimensional reduction based on isometry.
- Consistency problem in 4+d: the ansatz (31) with non-flat compact space is not a solution of the vacuum equations (p. 25); "early workers tended to ignore this".
- The review does not give the intermediate computation (inverse metric, Christoffels, Ricci of ansatz (5)); it quotes results from [10, 15, 41] and [8, 12, 13]. The tutorial must derive these independently.
- Riemann tensor convention is never written; only the Ricci contraction (4).
- Sign of g_44: (5) has +phi^2 while the ground state (21) has -1; the tutorial should fix g_44 = -phi^2 (spacelike fifth dimension in (+ - - -)) and re-check the signs of the F^2 and scalar terms in (9) rather than trust the printed sign in (5).
- Secs. 5-8 (projective, noncompactified/induced-matter, cosmology, astrophysics) are outside the tutorial's scope; only Sec. 6.1 (p. 30, timelike-signature remarks) and eq. (48) (p. 31) touch on reduction formulae.
