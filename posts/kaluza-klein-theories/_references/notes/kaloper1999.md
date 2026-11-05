# Identity

- arXiv: hep-th/9901045 (v2, 22 May 1999). Preprint numbers SU-ITP-98-19, McGill/98-30, dated January 1999.
- Title: "The O(dd) Story of Massive Supergravity".
- Authors: Nemanja Kaloper (Stanford), Robert C. Myers (McGill).
- Year: 1999. Venue: not stated in the PDF (only preprint numbers appear).
- Subject: generalised Scherk-Schwarz reductions of the low-energy heterotic string (10 -> D = 10 - d) on flat tori with internal fluxes and on "twisted tori" (group manifolds), assembled into an O(d, d+16)-covariant gauged (massive) supergravity.

# Structure

PDF page numbers first; printed page = PDF page - 1 (the title page is unnumbered).

1. Abstract, PDF 1.
2. Section 1 Introduction, PDF 2-4 (printed 1-3): motivation, the headline action eqs. (1)-(2), intuition for the non-abelian gauging, outline.
3. Section 2 A Review of Kaluza-Klein Reduction, PDF 4-9 (printed 3-8): action (3), ansatz (5)-(6), field redefinitions (7), reduced action (8)-(13), O(d,d+16) notation (14)-(18), origin of U(1)^{2d+16}.
4. Section 3 Reduction with Internal Fluxes, PDF 9-17 (printed 8-16): linear-in-y ansatz (19), constraint (20), zehnbein (21), reduced field strengths (22)-(25), reduced action (26)-(32), gauge transformations (33)-(38), gauge algebra (39)-(43), non-abelian field strength and Chern-Simons form (44)-(49), covariant potential (50)-(51), U-duality-invariant action (52)-(57).
5. Section 4 Twisted Tori, PDF 18-24 (printed 17-23): Bianchi II example, basis one-forms (58)-(59), Killing vectors and algebra (60)-(64), isometry transformations (65)-(69), invariant forms (70)-(71), metric ansatz (72)-(78), Yang-Mills ansatz (79)-(83), two-form ansatz (84)-(87).
6. Section 5 General Massive Reductions, PDF 25-30 (printed 24-29): reduced actions (88)-(97), gauge transformations (98)-(100), algebra (101)-(108), potential (109), Jacobi constraints (110)-(111), final action (112)-(116).
7. Section 6 Discussion, PDF 30-35 (printed 29-34): U-duality as pseudo-symmetry (117), quantisation of masses, gauge vs U-duality puzzle on T^3, moduli space, Stueckelberg nature of axions, validity regime of the effective action, classification of admissible internal manifolds (Bianchi algebras), relation to non-abelian KK reduction and Freund-Rubin.
8. Appendix A Reduction of Curvature and Dilaton, PDF 36-38 (printed 35-37): eqs. (118)-(134).
9. Appendix B Reduction of Yang-Mills Fields, PDF 38-41 (printed 37-40): eqs. (135)-(146).
10. Appendix C Reduction of Two-Form, PDF 41-44 (printed 40-43): eqs. (147)-(159).
11. Appendix D Gauge Algebra, PDF 44-50 (printed 43-49): eqs. (160)-(190).
12. Appendix E Generalized Axion Reductions of Ref. [6], PDF 50-52 (printed 49-51): eqs. (191)-(200), the T^3 axion ansatz and its linearisation by a matrix exponential.
13. References, PDF 52-55 (printed 51-54).

# Conventions

- Signature mostly plus, G = (-,+,+,...,+); curvature R^mu_{nu lambda sigma} = partial_lambda Gamma^mu_{nu sigma} - partial_sigma Gamma^mu_{nu lambda} + ... (printed p.4). String frame, alpha' = 1 (below eq. (4)). Fields are dimensionless, so the slopes/fluxes beta_{MNP}, m^I_{MN} have dimension length^{-1} and appear as masses (printed pp.8-9).
- Ten-dimensional action, eq. (3), printed p.4:
  $$S=\int d^{10}x\sqrt{-\mathcal G}\,e^{-\Phi}\Big\{\mathcal R+(\nabla\Phi)^2-\tfrac{1}{12}\mathcal H_{\mu\nu\lambda}\mathcal H^{\mu\nu\lambda}-\tfrac14\Sigma_{I=1}^{16}\mathcal F^I{}_{\mu\nu}\mathcal F^{I\mu\nu}\Big\},\qquad \mathcal H=d\mathcal B-\tfrac12\Sigma_I\mathcal A^I\wedge\mathcal F^I\ \ (4).$$
  Only the U(1)^{16} Cartan subalgebra of the Yang-Mills group is kept (footnote 1). D + d = 10. Caligraphic letters are 10d fields, upper-case Latin are D-dimensional; dilaton phi.
- Standard (untwisted) torus ansatz, eq. (5), printed p.4: dS^2 = g_{mu nu}dx^mu dx^nu + \hat{\mathcal G}_{MN}(dy^M + \hat V^M_mu dx^mu)(dy^N + \hat V^N_nu dx^nu), B and A^I decomposed likewise; standard KK means all hatted fields independent of y^M, with Phi(x,y) = phi(x) + (1/2) ln|det G_{MN}(x)| (eq. (6), printed p.5). Diagonalising redefinitions eq. (7). Reduced U(1)^{2d+16} multiplet A^a_mu = (V^M_mu, B_{mu M}, A^I_mu), eq. (15); moduli matrix M^{ab} of eq. (16) parametrising O(d,d+16)/O(d) x O(d+16); L^{ab} the off-diagonal O(d,d+16) metric, eq. (14).
- Scherk-Schwarz twist as defined by the paper (Introduction, printed p.1): "the dependence on the internal coordinates takes the form of a U-duality transformation that varies (in a specific way) over the internal space"; for an axionic (nilpotent) shift generator this is a linear dependence. Concretely the twist appears in three forms: (i) linear shifts of the axions by fluxes, eq. (19); (ii) y-dependent basis one-forms eta^M = N_N{}^M(y) dy^N on a group manifold ("twisted torus"), eqs. (58)-(62); (iii) the matrix form phi(x,y) = phi(x) exp(y B) of Appendix E, eqs. (196)-(198).

# Equations the tutorial uses

(a) Generalised reduction ansatz.
- Flux (axion-shift) ansatz on a flat T^d, eq. (19), printed p.8:
  $$\hat{\mathcal A}^I{}_M(x,y)=\mathcal A^I{}_M(x)+m^I_{MN}y^N,\quad \hat{\mathcal B}_{\mu M}(x,y)=\mathcal B_{\mu M}(x)+\tfrac12 m^I_{MN}y^N\big(A^I{}_\mu(x)+\mathcal A^I{}_P V^P{}_\mu(x)\big),\quad \hat{\mathcal B}_{MN}(x,y)=\mathcal B_{MN}(x)+\beta_{MNP}y^P+\mathcal A^I{}_{[M}(x)m^I_{N]P}y^P,$$
  all other fields as in (6). m^I_{MN} antisymmetric in MN, beta_{MNP} totally antisymmetric; constraint m^I_{[MN} m^I_{PQ]} = 0, eq. (20), from requiring dH to have no purely internal part on a flat torus.
- Twisted-torus (group-manifold) ansatz, eqs. (58), (59), (61), (62), (72), (79), printed pp.18-21:
  $$\mathcal E^A=\mathcal E^A{}_M(x)\,\eta^M(y,dy),\quad \eta^M=N_N{}^M(y)dy^N,\quad Z_M=N^N{}_M(y)\partial_N,\quad [Z_M,Z_N]=2\gamma^P_{MN}Z_P,\quad d\eta^M=-\gamma^M_{NP}\eta^N\wedge\eta^P,$$
  $$ds^2=g_{\mu\nu}(x)dx^\mu dx^\nu+\mathcal G_{MN}(x)\big(\eta^M(y)+V^M{}_\mu(x)dx^\mu\big)\big(\eta^N(y)+V^N{}_\nu(x)dx^\nu\big)\ (72),\qquad \mathcal A^I=\mathcal A^I{}_\mu(x)dx^\mu+\mathcal A^I{}_M(x)\eta^M+\sigma^I(y)\ (79),\ d\sigma^I=-m^I_{MN}\eta^M\wedge\eta^N\ (82).$$
  Jacobi gamma^M_{R[N}gamma^R_{PQ]} = 0 (63) and unimodularity gamma^N_{NM} = 0 (64) from invariance of the volume form.
- Matrix form phi(x,y) = phi(x) M(y): Appendix E, eq. (196)/(198), printed pp.50-51, for the three T^3 metric axions,
  $$\exp\begin{pmatrix}0&\mathcal A_1(x,y)&\bar{\mathcal A}_3(x,y)\\0&0&\mathcal A_2(x,y)\\0&0&0\end{pmatrix}=\exp\begin{pmatrix}0&\mathcal A_1(x)&\bar{\mathcal A}_3(x)\\0&0&\mathcal A_2(x)\\0&0&0\end{pmatrix}\exp\Big[y\begin{pmatrix}0&m_1&m_3\\0&0&m_2\\0&0&0\end{pmatrix}\Big]\ (198),$$
  i.e. the y-dependence is a one-parameter subgroup of the (nilpotent, upper-triangular) global symmetry acting on the axion matrix.

(b) Simplest explicit example and resulting mass/potential.
- Single-axion linear shift, eq. (192) (Appendix E, printed p.50, quoting ref. [6]): A_1(x,y) = A_1(x) + m_1 y, A_2(x,y) = A_2(x) + m_2 y, A_3(x,y) = A_3(x) + m_3 y + m_2 y A_1(x) + (1/2) m_1 m_2 y^2; after the redefinition A_3 = \bar A_3 + (1/2) A_1 A_2 (193) the ansatz becomes linear, eq. (195)/(200): \bar A_3(x,y) = \bar A_3(x) + m_3 y + (1/2) y (m_2 A_1(x) - m_1 A_2(x)). The paper does not reduce this example to a potential; it only shows y-independence of the field strengths (191).
- Flux example on a flat torus, mass terms: covariant derivatives eq. (25), printed p.10,
  $$\mathcal D_\mu\mathcal A^I{}_M=\partial_\mu\mathcal A^I{}_M-2m^I_{MN}V^N{}_\mu,\qquad \mathcal D_\mu\mathcal B_{MN}=\partial_\mu\mathcal B_{MN}+2m^I_{MN}A^I{}_\mu-3\beta_{MNP}V^P{}_\mu-2\mathcal A^I{}_{[M}m^I_{N]P}V^P{}_\mu,$$
  appearing in S_2 (28) as (1/2) G^{MN} D A^I_M D A^I_N + ... ; these are Stueckelberg couplings (printed p.31) giving the KK/Kalb-Ramond vectors masses of order m^I, beta.
- Resulting potential, eq. (32), printed p.11:
  $$W(\mathcal G,\mathcal A)=\tfrac34\mathcal G^{MN}\mathcal G^{PQ}\mathcal G^{RS}\big(\beta_{MPR}+2\mathcal A^I{}_{[M}m^I_{PR]}\big)\big(\beta_{NQS}+2\mathcal A^J{}_{[N}m^J_{QS]}\big)+\mathcal G^{MN}\mathcal G^{PQ}m^I_{MP}m^I_{NQ},$$
  independent of B_{MN}. With only Yang-Mills flux, W = G^{MN}G^{PQ} m^I_{MP} m^I_{NQ}. With a metric twist the extra terms are (eq. (89), printed p.24) -G_{MN}G^{PQ}G^{RS}gamma^M_{PR}gamma^N_{QS} - 2 G^{MN}gamma^P_{MQ}gamma^Q_{NP}.

(c) General statement: gauged supergravity with couplings = twist parameters.
- Structure constants eq. (103) (and (42) for fluxes only), printed p.27: f^M{}_{NP} = 2 gamma^M_{NP}, f^I{}_{MN} = f_{MN}{}^I = 2 m^I_{MN}, f_{MNP} = -3 beta_{MNP}; algebra eq. (104):
  $$[X^M,X^N]=[Y^I,Y^J]=[X^M,Y^I]=0,\ [X^M,Z_N]=2i\gamma^M_{NP}X^P,\ [Y^I,Z_M]=2im^I_{MN}X^N,\ [Z_M,Z_N]=-3i\beta_{MNP}X^P+2im^I_{MN}Y^I+2i\gamma^P_{MN}Z_P,$$
  with T_a = (Z_M, X^M, Y^I) the KK, Kalb-Ramond and Yang-Mills generators and Lie-algebra metric <T_a,T_b> = L_{ab} (Cartan metric degenerate). Coupling constant absorbed into f (printed p.15).
- Final action eq. (116) (= eq. (1)), printed p.28, and potential eq. (109) (= eq. (2)):
  $$S=\int d^Dx\sqrt{-g}e^{-\phi}\Big\{R+(\nabla\phi)^2+\tfrac18L_{ab}\mathcal D_\mu M^{bc}L_{cd}\mathcal D^\mu M^{da}-\tfrac14F^a_{\mu\nu}L_{ab}M^{bc}L_{cd}F^{d\mu\nu}-\tfrac1{12}H^2_{\mu\nu\lambda}-W(M)\Big\},$$
  $$W(M)=\tfrac1{12}M^{ad}M^{be}M^{cf}f_{abc}f_{def}-\tfrac14M^{ad}L^{be}L^{cf}f_{abc}f_{def},\qquad \mathcal D_\mu M^{ab}=\partial_\mu M^{ab}-f_{cd}{}^aA^c_\mu M^{db}-f_{cd}{}^bA^c_\mu M^{ad}\ (114),$$
  H = dB - (1/2) omega_CS with the non-abelian Chern-Simons form (107)-(108). Consistency constraints = Jacobi identity, eqs. (110)-(111).

(d) KK spectrum with twist: not present. The paper never writes shifted KK masses |n/R + m|; it only keeps the zero-mode truncation and estimates orders of magnitude (printed p.32): m_KK ~ 1/L, m_w ~ L/l_s^2, m_s ~ 1/l_s, versus flux masses beta ~ l_s^2/L^3, m^I ~ l_s/L^2 (from int_{3-cycle} H ~ n l_s^2, int_{2-cycle} F ~ n l_s) and geometric masses gamma^M_{NP} ~ L_M/(L_N L_P).

(e) Consistency: (i) the reduction ansatz must make the reduced field strengths independent of y^M and consistent with the 10d equations of motion, so that reduction and variation commute (printed pp.9, 20, 33); this yields the Jacobi conditions (20), (63), (64), (83), (87). (ii) Sufficient condition for a finite-dimensional truncation (printed p.33, citing [34]): pullbacks of the isometry generators span a complete, linearly independent, orthogonal basis on the tangent space, i.e. the internal manifold is homogeneous with isometry algebra of dimension equal to its dimension, and unimodular (64). In d = 2 only the abelian algebra survives; in d = 3 Bianchi I, II, VIII, VI(h = -1), IX (printed p.33). (iii) Residual gauge transformations must themselves be y-independent (Appendix D, printed p.43). (iv) Validity of the effective action additionally requires the induced masses to lie below the KK, winding and string scales (printed p.32); for geometric twists this needs anisotropic radii.

(f) T-duality/O(d,d): the standard T^d reduction has global O(d,d+16,R) (14)-(18), of which O(d,d+16,Z) is the exact string T-duality group [29]. In the twisted reduction the mass parameters assemble into the totally antisymmetric f_{abc} = f_{[ab}{}^d L_{d|c]} and the action (116) remains O(d,d+16) covariant provided the duality also acts on the couplings, f_{abc} -> Omega_a{}^d Omega_b{}^e Omega_c{}^f f_{def} (57), (117). Thus U-duality maps one reduced massive theory to another (spontaneous breaking by topological background fluxes/curvatures), and is a pseudo-symmetry in the sense of Hull-van Proeyen [28] (printed p.29). The gauge group is a non-semisimple subalgebra embedded in the O(d,d+16) algebra (printed p.14); the global gauge transformations H further quotient the moduli space (printed p.31). Quantisation of fluxes and SL(d,Z) twists break O(d,d+16,R) to O(d,d+16,Z), conjectured exact (printed pp.29-30).

# Attributable statements

1. Scherk-Schwarz reduction = internal-coordinate dependence through a global (U-duality) symmetry transformation varying over the internal space; for a nilpotent axionic shift generator it is linear in y (Section 1, printed p.1).
2. The slope parameters of the linear dependence become mass parameters of the reduced theory, while the reduced action stays y-independent (Section 1, printed p.1; Section 3, printed pp.8-9, eq. (19)).
3. Non-commuting axionic shifts cannot all be turned on with a linear ansatz; general Scherk-Schwarz needs polynomial (or more general) y-dependence, which can be linearised by a field redefinition amounting to a functional logarithm (Section 1, printed p.1; Appendix E, eqs. (196)-(200), printed pp.50-51).
4. Mass parameters play a dual role: couplings in the scalar potential and structure constants of the non-abelian gauge group, so the twisted reduction yields a gauged supergravity (Section 1 eqs. (1)-(2); Section 5 eqs. (103)-(109), printed pp.27-28).
5. Intuition for the non-abelian gauging: KK gauge transformations are local internal shifts, which with a y-dependent axion must be accompanied by local axionic shifts, promoting part of the global duality group to a local one (Section 1, printed p.2).
6. The twisted-torus (group-manifold) reduction is a non-abelian KK reduction on a homogeneous space with transitive isometry group of the same dimension; consistency and finiteness of the truncation are related, and unimodularity gamma^N_{NM} = 0 is required (Section 4, eqs. (61)-(64), printed pp.18-19; Section 6, printed p.33).
7. The axions become Stueckelberg fields; the generalised reduction reduces, rather than enhances, the number of massless modes (Section 6, printed p.31).
8. U-duality remains a symmetry of the full theory but is spontaneously broken by the topological background; it acts on the couplings as f_{abc} -> Omega Omega Omega f and maps one reduced theory to another (Section 3 eq. (57), printed p.16; Section 6 eq. (117), printed p.29).
9. The constant global gauge transformation and the U-duality transformation producing the same shift tau -> tau + 1 of the T^2 modulus are distinct ten-dimensional symmetries (the former includes a translation around the S^1) (Section 6, printed p.30).
10. Scherk-Schwarz reduction can be viewed as an application of non-abelian KK reduction on a group manifold admitting finite-dimensional truncation (Section 6, printed p.34).

# Caveats

- No explicit single-circle (S^1) worked example with a scalar phi -> phi + m y and a derived potential/mass is given; the T^3 metric-axion ansatz of Appendix E is quoted from ref. [6] (Kaloper-Khuri-Myers 1998) without reducing the action. The tutorial should use eq. (19)+(25)+(32) or Appendix E, not claim a worked circle reduction from this paper.
- No KK tower or shifted-mass spectrum appears anywhere; only zero-mode truncations and order-of-magnitude scales (printed p.32).
- Factor conventions: structure constants of the isometry algebra are defined with a factor 2, [Z_M,Z_N] = 2 gamma^P_{MN} Z_P (61), and the gauge-algebra generators carry an i, [T_a,T_b] = i f_{ab}{}^c T_c (40), (102); the Lie-algebra metric used is L_{ab}, not the (degenerate) Cartan metric.
- The potential (2)/(109) contains a term linear in M absent from the pure-flux potential (50); it comes from the gamma-gamma curvature terms of (89).
- Only the U(1)^{16} Cartan part of the heterotic gauge group is kept; Wilson lines are assumed to break the rest (footnote 1). Group manifolds without non-trivial 1-cycles would keep the full non-abelian gauge symmetry (printed p.34).
- Heterotic string frame with e^{-phi} prefactor; the dilaton shift (6)/(90) uses sqrt(det G), and det G_{MN} is a gauge singlet even with twists (printed p.24).
- Flux/twist quantisation (Dirac, Rohm-Witten, SL(d,Z)) is only discussed qualitatively (printed pp.29-30). The semisimple cases (Bianchi IX = SU(2), S^3) give geometric masses of order 1/L and the truncated action is not trustworthy as it stands (printed p.33).
- Not all claims are proved in the text: the ansatz (19) is "merely quoted"; full derivations are deferred to Appendices A-D.
