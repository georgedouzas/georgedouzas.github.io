# Identity

- arXiv: hep-ph/0010141 (v2, 9 Dec 2000); preprint number NTUA-96/00.
- Title: "Supersymmetry Breaking by Dimensional Reduction over Coset Spaces".
- Authors: P. Manousselis and G. Zoupanos (Physics Department, National Technical University of Athens).
- Year: 2000. No journal venue stated on the PDF (preprint only). 13 PDF pages (title page unnumbered, then printed pages 1-12).

# Structure

PDF page = printed page + 1 (title page is PDF p.1, unnumbered).

1. Title/abstract — PDF p.1.
2. Sec. 1 "Introduction" — PDF pp.2-3 (printed 1-2). Motivation (SUSY breaking, TeV strings, E8xE8), one-paragraph description of CSDR idea.
3. Sec. 2 "The Coset Space Dimensional Reduction" — PDF pp.3-7 (printed 2-6). Metric ansatz, coset algebra (1), Maurer-Cartan (2), D-dim action (3)-(6), symmetry conditions (7)-(9), rules for H and scalars (10)-(11), fermion rules (12)-(15), anomaly condition (16), 4D Lagrangian (17), potential (18), constraint (19), Yukawa/torsion (20)-(25).
4. Sec. 3 "Supersymmetry Breaking by Dimensional Reduction over Symmetric Coset Spaces" — PDF pp.7-8 (printed 6-7). Example SO(7)/SO(6)=S^6 with G=E8, eq. (26); theorems (i),(ii) on spontaneous breaking.
5. Sec. 4 "Supersymmetry Breaking by Dimensional Reduction over Non Symmetric Coset Spaces" — PDF pp.8-11 (printed 7-10). Example G2/SU(3) with G=E8, eqs. (27)-(41); explicit potential and soft terms.
6. Sec. 5 "Conclusion" — PDF p.11 (printed 10).
7. References [1]-[27] — PDF pp.12-13 (printed 11-12). Key CSDR refs: [12] Forgacs-Manton 1980, Witten 1977; [13] Kapetanakis-Zoupanos Phys. Rept. C219 (1992); [14] Kubyshin et al. LNP 349.

# Conventions

- Spacetime: D-dimensional M^D compactified to M^4 x S/R, D = 4 + d, d = dim S - dim R. Metric block diagonal (Sec. 2, PDF p.3):
  g^{MN} = diag(eta^{mu nu}, -g^{ab}), with eta^{mu nu} = diag(1,-1,-1,-1) ("mostly minus"), g^{ab} the coset metric. Note the explicit minus sign in front of g^{ab}.
- Coordinates z^M = (x^m, y^alpha); alpha = curved coset index, a = tangent-space coset index; y defines a coset representative L(y) in S.
- Generators of S: Q_A, A = 1..dim S, split into Q_i of R (i = 1..dim R) and Q_a of S/R (a = dim R + 1..dim S). Commutators, eq. (1), PDF p.3:
  [Q_i, Q_j] = f_{ij}^k Q_k,  [Q_i, Q_a] = f_{ia}^b Q_b,  [Q_a, Q_b] = f_{ab}^i Q_i + f_{ab}^c Q_c.
  Index placement: lower pair = the two commuted generators, upper = result. S/R symmetric iff f_{ab}^c = 0. No explicit "i" factor in (1) (anti-Hermitian-type convention); contrast with eq. (30) for G2 which uses [Q^a,Q^b] = 2 i f^{abc} Q^c.
- Vielbein/connection from the Maurer-Cartan form, eq. (2), PDF p.3: L^{-1}(y) dL(y) = e^A_alpha Q_A dy^alpha. At the origin y = 0 with L(y) = exp(y^a Q_a): e^a_alpha = delta^a_alpha and e^i_alpha = 0 (PDF p.4).
- Killing metric / radii: not written as a general formula. In the G2/SU(3) example (PDF p.11) the "most general G2-invariant metric on G2/SU(3)" is stated to be g_{ab} = r delta_{ab} (a single radius r); the paper also says T_{abc} is proportional to f_{abc} "when the radii are equal" (PDF p.7), referring to [13] for multi-radius cases. Gamma matrices (PDF p.11): [Gamma^mu, Gamma^nu] = 2 eta^{mu nu}, Gamma^mu = gamma^mu x I_8; [Gamma^a, Gamma^b] = -2 g^{ab}, Gamma^a = (1/sqrt r) gamma_5 x [[0, gbar^a],[gbar^a, 0]] (a = 1,2,3,5,6), Gamma^4 = (1/sqrt r) gamma_5 x [[0, i I_4],[i I_4, 0]]; gbar^1 = sigma^1 x sigma^2, gbar^2 = sigma^2 x sigma^2, gbar^3 = -I_2 x sigma^3, gbar^5 = sigma^3 x sigma^2, gbar^6 = -I_2 x sigma^1; Sigma^{ab} = (1/4)[Gamma^a, Gamma^b]. (The paper writes these as anticommutator-like relations with square brackets; it is clearly the Clifford relation.)
- Gauge field: A_M, anti-Hermitian-style convention with NO coupling constant: D_M = partial_M - theta_M - A_M (eq. 4), theta_M = (1/2) theta_{M N Lambda} Sigma^{N Lambda} spin connection (eq. 5), F_{MN} = partial_M A_N - partial_N A_M - [A_M, A_N] (eq. 6). No gauge coupling g appears anywhere; the trace normalisation is given per example (Tr Q^a Q^b = 2 delta^{ab} etc.).
- Higher-dimensional action, eq. (3), PDF p.4:
  A = int d^4x d^d y sqrt(-g) [ -(1/4) Tr(F_{MN} F_{K Lambda}) g^{MK} g^{N Lambda} + (i/2) psibar Gamma^M D_M psi ].
- Group-theory notation: G = higher-dim gauge group; R_G = image of R in G; H = C_G(R_G) the centraliser; K = C_G(S) when S embeds in G.

# Equations the tutorial uses

## (a) Symmetry condition and CSDR constraints

Eqs. (7)-(9), PDF p.4. With g(s), f(s) gauge transformations in adjoint and F reps of G corresponding to s in S acting on S/R:
  A_mu(x, y) = g(s) A_mu(x, s^{-1} y) g^{-1}(s),                                   (7)
  A_alpha(x, y) = g(s) J_alpha^beta A_beta(x, s^{-1} y) g^{-1}(s) + g(s) partial_alpha g^{-1}(s),   (8)
  psi(x, y) = f(s) Omega psi(x, s^{-1} y) f^{-1}(s),                               (9)
where J_beta^alpha is the Jacobian of s and Omega is the Jacobian plus a local Lorentz rotation in the tangent space (fermions). Note the paper writes f^{-1}(s) on the right of psi in (9) as printed.

Resulting algebraic constraint on the scalars, eq. (19), PDF p.6:
  f_{ai}^D phi_D - [phi_a, phi_i] = 0,
where phi_i generate R_G (i.e. the constraint says phi_a intertwines the R-action on S/R with the R_G-action in G). The constraints "imply that some components phi_a are zero, some are constants and the rest can be identified with the genuine Higgs fields" (PDF p.6). The fermion constraint is stated in words: "the CSDR constraints tell us that partial_a psi = 0" (PDF p.7). The paper does NOT write the constraint for A_mu as a displayed equation; it states A_mu must be y-independent and commute with R_G (PDF p.4).

## (b) Rules for the 4D gauge group and scalar content

- Gauge group (PDF p.4, no equation number): "the four-dimensional gauge group H is the centralizer of R in G, H = C_G(R_G)."
- Scalars: phi_alpha := A_alpha transform under R as the vector v, eq. (10), PDF p.4:
    S ⊃ R,  adj S = adj R + v.                                                   (10)
  Decompose, eq. (11), PDF p.5:
    G ⊃ R_G x H,  adj G = (adj R, 1) + (1, adj H) + sum (r_i, h_i).               (11)
  Rule (PDF p.5): if v = sum s_i (irreps of R), "there survives an h_i multiplet for every pair (r_i, s_i), where r_i and s_i are identical irreducible representations of R." Justified via Schur's lemma, phi_alpha being an intertwiner.

## (c) Reduced 4D Lagrangian and potential

Eq. (17), PDF p.6 (C = volume of the coset space; t = gauge index of H):
  A = C int d^4x ( -(1/4) F^t_{mu nu} F^{t mu nu} + (1/2) (D_mu phi_alpha)^t (D^mu phi^alpha)^t + V(phi) + (i/2) psibar Gamma^mu D_mu psi - (i/2) psibar Gamma^a D_a psi ),
with D_mu = partial_mu - A_mu, D_a = partial_a - theta_a - phi_a, theta_a = (1/2) theta_{abc} Sigma^{bc}. Note the sign of V(phi) as printed is +V inside the Lagrangian (with V defined negative-definite-looking in (18)); the paper does not comment.

Potential, eq. (18), PDF p.6 (the "formal" expression before solving constraints):
  V(phi) = -(1/4) g^{ac} g^{bd} Tr( f_{ab}^C phi_C - [phi_a, phi_b] ) ( f_{cd}^D phi_D - [phi_c, phi_d] ),
with A, C, D = 1..dim S (so f_{ab}^C phi_C = f_{ab}^i phi_i + f_{ab}^c phi_c) and f the structure constants of (1). No explicit separate radii factors appear beyond g^{ac} g^{bd}; the dependence on f_{ai}^b enters only through the constraint (19). The paper then states (PDF p.6): "When V(phi) is expressed in terms of the unconstrained independent Higgs fields, it remains a quartic polynomial which is invariant under gauge transformations of the final gauge group H, and its minimum determines the vacuum expectation values of the Higgs fields [22]." No general closed form of V in terms of f_{ai}^b, radii, or R-generators is written; see the G2/SU(3) example (32), (36) for an explicit case.

Theorems on the minimum (PDF p.8, Sec. 3, quoted from [13],[25]):
  (i) "When S/R is symmetric, the form of the potential of the four-dimensional theory is such that it necessarily leads to a spontaneous breakdown of H";
  (ii) "When S has an isomorphic image in G, the four dimensional gauge group H always breaks down spontaneously to a subgroup K which is the centralizer of S in G."

## (d) Fermion constraint and rules (brief)

Eqs. (12)-(15), PDF p.5. Decompose F (fermion rep of G) under R_G x H, F = sum (t_i, h_i) (12), and the SO(d) spinor under R, sigma_d = sum sigma_j (13); an h_i multiplet survives for each pair of identical R-irreps (t_i, sigma_i). Weyl condition in D = 4n+2: sigma_D = (2,1,sigma_d) + (1,2, sigmabar_d) (14), with sigma_d = sum sigma_k, sigmabar_d = sum sigmabar_k (15). Majorana+Weyl compatible in D = 4n+2; F real needs D = 2+8n, pseudoreal D = 6+8n (PDF p.6). Anomaly-freedom for G = E8 in D = 10, eq. (16): l(G) = 60, with l(G) the sum of R-indices in 248 under E8 ⊃ R x H, normalised so the vector v of (10) has index two.
Yukawa term, eqs. (20)-(22), PDF pp.6-7: L_Y = (i/2) psibar Gamma^a nabla_a psi + psibar V psi with nabla_a = -partial_a + (1/2) f_{ibc} e^i_gamma e^gamma_a Sigma^{bc} + phi_a (21), V = (i/4) Gamma^a G_{abc} Sigma^{bc} (22). Torsionful connection theta^a_{cb} = -f^a_{ib} e^i_alpha e^alpha_c - G^a_{cb} with G^a_{cb} = D^a_{cb} + (1/2) T^a_{cb} (23), D^a_{cb} = g^{ad} (1/2)[ f^e_{db} g_{ec} + f^e_{cb} g_{de} - f^e_{cd} g_{be} ] (24), T_{abc} = 2 tau (D_{abc} + D_{bca} - D_{cba}) (25), tau free. At y = 0, e^i_gamma = 0 so nabla_a = phi_a. V anticommutes with the 6D helicity operator and commutes with T_i = -(1/2) f_{ibc} Sigma^{bc}; by Schur only R-singlets common to 4 and 4bar of SO(6) get the geometric mass; V vanishes for symmetric cosets since f_{ab}^c = 0 (PDF p.7).

## (e) Explicit coset examples

No SU(2)/U(1) or S^2 example appears anywhere in the paper; both examples are six-dimensional cosets with G = E8, D = 10.

Example 1 (Sec. 3, PDF pp.7-8): S/R = SO(7)/SO(6) = S^6 (symmetric), G = E8, Weyl-Majorana fermions in the adjoint 248. Embedding eq. (26):
  E8 ⊃ SO(6) x SO(10),  248 = (15,1) + (1,45) + (6,10) + (4,16) + (4bar,16bar).
R = SO(6) content of vector and spinor of S^6: 6 and 4. Anomaly condition (16) satisfied. H = C_{E8}(SO(6)) = SO(10); scalars: one 10 of SO(10); fermions: one left-handed 16 of SO(10). No supersymmetry survives. By theorems (i),(ii) SO(10) breaks spontaneously to K = C_{E8}(SO(7)) = SO(9). The potential is NOT written explicitly for this example.

Example 2 (Sec. 4, PDF pp.8-11): S/R = G2/SU(3) (non-symmetric), G = E8. Embedding eq. (27):
  E8 ⊃ SU(3) x E6,  248 = (8,1) + (1,78) + (3,27) + (3bar,27bar).
R = SU(3) content of G2/SU(3) vector 3 + 3bar and spinor 1 + 3. H = C_{E8}(SU(3)) = E6, fermions in 78 and 27, scalars in 27 (N=1 SUSY E6 theory with soft breaking).
G2 ⊃ SU(3), 14 = 8 + 3 + 3bar (28); generators Q_{G2} = {Q^a, Q^rho, Q_rho} (29), a = 1..8, rho = 1,2,3. Commutators eq. (30), PDF p.9:
  [Q^a, Q^b] = 2 i f^{abc} Q^c,  [Q^a, Q^rho] = -(lambda^a)^rho_sigma Q^sigma,  [Q^rho, Q_sigma] = -(lambda^a)^rho_sigma Q^a,  [Q^rho, Q^sigma] = 2 sqrt(2/3) epsilon^{rho sigma tau} Q_tau,
normalisation Tr Q^a Q^b = 2 delta^{ab}, Tr Q^rho Q_sigma = 2 delta^rho_sigma (unnumbered line after (30)). Only the "non trivial" commutators are listed; [Q^a, Q_rho] and [Q_rho, Q_sigma] are not written.
Fields {phi_i, phi^rho, phi_rho} (31), phi_i equal to the R generators. Potential of ANY theory reduced over G2/SU(3), eq. (32), PDF p.9 (from [13]):
  V(phi) = 8 + (4/3) Tr(phi^rho phi_rho) - (1/2) (lambda^i)^rho_sigma Tr( J_i [phi_rho, phi^sigma] ) + sqrt(2/3) epsilon^{rho sigma tau} Tr( phi_tau [phi_rho, phi_sigma] ) - (1/4) Tr( [phi_rho, phi_sigma][phi^rho, phi^sigma] + [phi^rho, phi_sigma][phi_rho, phi^sigma] ).
E8 generators Q_{E8} = {Q^a, Q^alpha, Q^{i rho}, Q_{i rho}} (33), a = 1..8, alpha = 1..78, i = 1..27, rho = 1,2,3; Tr Q^a Q^b = 2 delta^{ab}, Tr Q^alpha Q^beta = 12 delta^{alpha beta}, Tr Q^{i rho} Q_{j sigma} = 2 delta^i_j delta^rho_sigma. Commutators eq. (34), PDF p.10: [Q^a,Q^b] = 2 i f^{abc} Q^c, [Q^alpha,Q^beta] = 2 i g^{alpha beta gamma} Q^gamma, [Q^a, Q^{i rho}] = -(lambda^a)^rho_sigma delta^i_j Q^{j sigma}, [Q^{i rho}, Q^{j sigma}] = (1/sqrt 6) epsilon^{rho sigma tau} d^{ijk} Q_{k tau}, [Q^{i rho}, Q_{j sigma}] = -(lambda^a)^rho_sigma delta^i_j Q^a + (1/6) delta^rho_sigma (G^alpha)^i_j Q^alpha, [Q^alpha, Q^{i rho}] = (G^alpha)^i_j delta^rho_sigma Q^{j sigma}; d^{ijk}, (G^alpha)^i_j from Kephart-Vaughn [27].
Solution of constraints (19), eq. (35): phi^a = Q^a, phi_rho = beta^i Q_{i rho}, phi^rho = beta_i Q^{i rho}. Potential in terms of the genuine Higgs beta (27 of E6), eq. (36), PDF p.10:
  V(beta) = 8 - (40/3) beta^2 - 4 d_{ijk} beta^i beta^j beta^k + beta^i beta^j d_{ijk} d^{klm} beta_l beta_m + (11/4) sum_alpha beta^i (G^alpha)^j_i beta_j beta^k (G^alpha)^l_k beta_l.
Superpotential W(B) = (1/3) d_{ijk} B^i B^j B^k (37); D-terms D^alpha = sqrt(11/2) beta^i (G^alpha)^j_i beta_j (38); soft terms L_SSB = -(40/3) beta^2 - 4 d_{ijk} beta^i beta^j beta^k (39). D_{abc} = (1/2) f_{abc} (40), G_{abc} = (1/2)(1 - 3 tau) f_{abc} (41); gaugino geometric mass (1 - 3 tau) 6/sqrt 3 (PDF p.11). Metric used: g_{ab} = r delta_{ab}; the numerical coefficients in (32),(36) are quoted with r set to 1 (the paper says soft terms "depend only on the radius" but does not display r).

# Attributable statements

1. (Sec. 1, PDF p.2) CSDR: "impose the condition that a symmetry transformation by an element of the isometry group S of B corresponds to a gauge transformation. Then the Lagrangian will be independent of the extra coordinates just because it is gauge invariant."
2. (Sec. 1, PDF p.2-3) Fields obtained by CSDR "are the first terms in the expansion of the D-dimensional fields in harmonics of the internal space B and are massless after the first stage of the symmetry breaking."
3. (Sec. 2, PDF p.3) Metric ansatz g^{MN} = diag(eta^{mu nu}, -g^{ab}) with eta = diag(1,-1,-1,-1); generators split Q_i (R), Q_a (coset), eq. (1); S/R symmetric iff f_{ab}^c = 0.
4. (Sec. 2, PDF p.4) Symmetry conditions (7)-(9); A_mu become y-independent 4D gauge fields commuting with R_G, hence H = C_G(R_G).
5. (Sec. 2, PDF pp.4-5) Scalar rule: adj S = adj R + v (10), adj G = (adj R,1)+(1,adj H)+sum(r_i,h_i) (11); surviving h_i for each identical pair (r_i, s_i), by Schur's lemma.
6. (Sec. 2, PDF p.6) 4D Lagrangian (17) with volume factor C, potential (18) V = -(1/4) g^{ac} g^{bd} Tr(f_{ab}^C phi_C - [phi_a,phi_b])(f_{cd}^D phi_D - [phi_c,phi_d]), and constraint (19) f_{ai}^D phi_D - [phi_a, phi_i] = 0.
7. (Sec. 2, PDF p.6) After solving the constraints V "remains a quartic polynomial which is invariant under gauge transformations of the final gauge group H, and its minimum determines the vacuum expectation values of the Higgs fields."
8. (Sec. 3, PDF p.8) Theorems: symmetric S/R forces spontaneous breakdown of H; if S embeds in G, H breaks to K = C_G(S).
9. (Sec. 2, PDF p.7) For non-symmetric cosets the geometric (torsion) term V = (i/4) Gamma^a G_{abc} Sigma^{bc} gives large masses to R-singlet fermions (gauginos); absent for symmetric cosets since f_{ab}^c = 0.
10. (Sec. 3-4, PDF pp.7-8) E8 over S^6 = SO(7)/SO(6) gives SO(10) with a 10 and a 16, SUSY fully broken; E8 over G2/SU(3) gives softly broken N=1 E6 with 78, 27.

# Caveats

- This is a short research letter; Sec. 2 is a compressed review and cites [13] (Kapetanakis-Zoupanos 1992) for derivations. Several key formulas (general V in terms of radii and f_{ai}^b, the explicit solution of (19), the Killing-metric normalisation) are not derived here.
- No SU(2)/U(1), S^2, or any four-dimensional-from-six-dimensional example; both examples are 10D -> 4D over 6D cosets. The tutorial cannot cite this paper for an explicit S^2 potential.
- No gauge coupling constant anywhere: A_M absorbs g; F_{MN} = partial A - partial A - [A, A] (minus sign on the commutator, anti-Hermitian convention). Trace normalisations are example-specific (Tr Q^a Q^b = 2 delta^{ab}, Tr Q^alpha Q^beta = 12 delta^{alpha beta}).
- Sign/placement of V in (17): the potential appears as +V(phi) in the Lagrangian with V in (18) carrying an overall -1/4; no comment is made about the mostly-minus signature compensating. Users should re-derive before quoting the overall sign.
- Eq. (18) uses capital C, D summed over all of S (A = 1..dim S), so f_{ab}^C phi_C contains both f_{ab}^i phi_i and f_{ab}^c phi_c; phi_i are then fixed by the constraint to be the R_G generators (phi_i "generate R_G", PDF p.6; eq. (35) phi^a = Q^a in the G2 example).
- The coefficients in (32), (36) (8, 4/3, 40/3, 11/4, sqrt(2/3), 1/sqrt 6) presuppose the specific normalisations of (30), (34) and g_{ab} = r delta_{ab} with r effectively 1; no radius appears explicitly despite the text saying the soft terms depend on the radius.
- The metric g_{ab} = r delta_{ab} is asserted as the "most general G2-invariant metric on G2/SU(3)" (PDF p.11).
- Several typos as printed: "disccard", "posses", "supesymmetric"; in (9) the ordering f(s) Omega psi f^{-1}(s) should be read with care. Equation numbering: (1)-(41) only; the gamma-matrix formulas and the gaugino mass (1-3 tau) 6/sqrt 3 on PDF p.11 are unnumbered, as are the normalisation lines after (30) and (33).
- The torsion parameter tau in (25) is free; the gaugino mass depends on it, while scalar soft terms do not.

# Sections and equations used (fermion parts, task 12.7)

part-v-02-four-dimensional-theory (sec-csdr-fermions):
- Eq. (9) (symmetry condition on spinors): def-csf-symmetric-spinor, exr-csf-infinitesimal.
- Eqs. (12)-(15) (fermion rule, Weyl condition in D = 4n + 2): prp-csf-fermion-constraint, prp-csf-chiral-content,
  exr-csf-matching-rule, exr-csf-dirac-six. Eqs. (20)-(21) (Yukawa terms): prp-csf-chiral-content. Eq. (16) (anomaly
  condition for E_8): exr-csf-anomaly.

part-vi-02-compactifications (sec-compactification-fermions):
- Sec. 3, eq. (26) (E_8 over SO(7)/SO(6): SO(10), one 16, one 10, no supersymmetry): exm-cff-s6-symmetric.
- Sec. 4, eqs. (27)-(28) (E_8 over G_2/SU(3): spinor 1 + 3, fermions in 78 and 27): prp-cff-nk-fermions,
  exm-cff-g2su3, exr-cff-multiplets.
- Eq. (22) and PDF p. 7 (geometric term V, commutes with R, Schur, vanishes for symmetric cosets), eqs. (40)-(41) and
  the gaugino mass (1 - 3 tau) 6/sqrt 3 of PDF p. 11: def-cff-geometric-mass, exm-cff-gaugino-mass,
  exm-cff-symmetric-no-mass, exr-cff-schur, exr-cff-tau.
