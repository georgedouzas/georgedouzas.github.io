# Identity

- arXiv: 1602.03673 [hep-th] (v1, 11 Feb 2016)
- Title: Higher-Dimensional Unified Theories with continuous and fuzzy coset spaces as extra dimensions
- Authors: G. Manolakos (NTU Athens), G. Zoupanos (NTU Athens and ITP Heidelberg)
- Year: 2016
- Venue: footnote on p. 1: "Based on a talk given by G.Z. as invited main speaker at the 11-th International
  Workshop on 'Lie Theory and Its Applications in Physics' (LT-11), 15-21 June 2015, Varna, Bulgaria, to be
  published in Springer." (Springer proceedings volume; no volume/page given in the PDF.)
- Length: 20 pages; printed page numbers coincide with the PDF page numbers shown by the Read tool.

# Structure

1. Introduction — pp. 1-2.
2. The coset space dimensional reduction of a D-dimensional YMD Lagrangian — pp. 2-5 (eqs. (1)-(15)).
   This is the only part the tutorial needs. It contains: the symmetric-field constraints (2)-(4), the CSDR rules
   (5)-(8), the reduced action (9)-(11), the scalar constraint (12), the vacuum criterion (13), and the summary of the
   E8 on SU(3)/U(1)xU(1) model (14)-(15).
3. Fuzzy spaces — pp. 5-8. 3.1 The fuzzy sphere, pp. 5-7 (eqs. (16)-(40)); 3.2 Gauge theory on the fuzzy sphere,
   p. 8 (eqs. (41)-(49)). NOT needed by the tutorial.
4. Ordinary fuzzy dimensional reduction and gauge symmetry enhancement — pp. 8-9 (eqs. (50)-(54)). NOT needed.
5. Fuzzy CSDR — pp. 9-11 (eqs. (55)-(72)). NOT needed, but (59)-(61) are the fuzzy analogue of the continuous
   constraints and are recorded below for comparison.
6. Orbifolds and fuzzy extra dimensions — pp. 11-16. 6.1 N=4 SYM field theory and Z3 orbifolds, pp. 12-13
   (eqs. (73)-(77)); 6.2 Dynamical generation of twisted fuzzy spheres, pp. 13-14 (eqs. (78)-(90)); 6.3 Chiral models
   after the fuzzy orbifold projection: the SU(3)_c x SU(3)_L x SU(3)_R model, pp. 15-16 (eqs. (91)-(96)). NOT needed.
7. Unnumbered closing paragraphs (outlook) and Acknowledgement — p. 16. References [1]-[81] — pp. 16-20.

# Conventions

- Signature: mostly minus. After compactification M^D -> M^4 x S/R the metric is written (eq. (9), p. 4)
  g^{MN} = diag(eta^{mu nu}, -g^{ab}) with eta^{mu nu} = diag(1, -1, -1, -1) and g^{ab} "the metric of the coset".
  Hence the coset block g^{ab} is positive-definite and enters with a minus sign.
- Indices: M, N = 1, ..., D (D-dimensional); mu, nu four-dimensional; alpha, beta = dim R + 1, ..., dim S are the
  CURVED coset indices; a, b, c, d are used both for flat (tangent) coset indices and as labels of the coset
  generators of S; i labels the generators of R; A, B, C, D = 1, ..., dim S label all generators of S. The paper
  does not introduce separate symbols Q_A for the generators; the Lie algebra of S enters only through its
  structure constants and through the Killing vectors xi_A^alpha.
- Structure constants: written f^C_{AB} (upper index third), "the structure constants appearing in the commutators
  of the Lie algebra of S" (p. 4, after (11)). The paper never writes the commutator explicitly; from the fuzzy
  analogue (63), [T^a, T^b] = C^{ab}_c T^c, and from (12) the convention is [Q_A, Q_B] = f^C_{AB} Q_C with the
  coset/R split f^D_{ai} (one R index, one coset index) and f^c_{ab} (all coset). No normalisation of the generators
  and no Killing metric are specified. No coset radii R_a appear anywhere; the coset metric is left as a generic
  g^{ab} and the volume of S/R is the constant C in (10).
- Symmetric coset: footnote 1, p. 4: "A coset space is called symmetric when f^c_{ab} = 0."
- Gauge field: anti-hermitian Lie-algebra-valued, coupling absorbed (no g appears in Section 2). Covariant derivative
  D_M = partial_M - theta_M - A_M with theta_M = (1/2) theta_{M N Lambda} Sigma^{N Lambda} the spin connection of M^D;
  field strength F_{MN} = partial_M A_N - partial_N A_M - [A_M, A_N] (p. 3, below (1)). So delta A = dW - [W, A] is a
  gauge transformation (cf. (3)), i.e. A -> A + D W with D W = dW - [W, A].
- Higher-dimensional action (eq. (1), p. 3; D-dimensional YM with gauge group G coupled to fermions):
  A = int d^4x d^d y sqrt(-g) [ -(1/4) Tr(F_{MN} F_{K Lambda}) g^{MK} g^{N Lambda} + (i/2) bar-psi Gamma^M D_M psi ].
  Fermions can be in any representation F of G "unless an additional symmetry, e.g. supersymmetry, is considered".
- Killing vectors: xi_A^alpha, A = 1, ..., dim S, generate the S-isometries of S/R; W_A is the compensating gauge
  transformation associated with xi_A and depends only on the internal coordinates y; D(W_A) denotes the gauge
  transformation in the representation to which the field belongs (p. 3).
- Reduced fields: A_M -> (A_mu, A_a); A_alpha(x, y) = phi_alpha(x, y) are the 4D scalars. The superscript t in (10)
  is the adjoint (gauge-generator) component index of the 4D fields.
- Mapping to the tutorial: the tutorial uses hermitian generators and an explicit coupling; this paper's F = dA - dA
  - [A, A] with anti-hermitian A corresponds to A_here = -i g A_tut (so that D = partial - A_here = partial - i g
  A_tut). Signs of the potential depend on the trace convention (see Caveats).

# Equations the tutorial uses

(a) Symmetry condition (symmetric fields), eqs. (2)-(4), p. 3. The S-transformations of the fields are gauge
transformations:

$$\delta_A \phi = \xi_A^\alpha \partial_\alpha \phi = D(W_A)\phi, \tag{2}$$

$$\delta_A A_\alpha = \xi_A^\beta \partial_\beta A_\alpha + \partial_\alpha \xi_A^\beta A_\beta
 = \partial_\alpha W_A - [W_A, A_\alpha], \tag{3}$$

$$\delta_A \psi = \xi_A^\alpha \partial_\alpha \psi - \tfrac12 G_{Abc}\Sigma^{bc}\psi = D(W_A)\psi, \tag{4}$$

"where W_A depend only on internal coordinates y and D(W_A) represents a gauge transformation in the corresponding
representation where the fields belong. Solving the above constraints (2)-(4), we result with [3,4] the
unconstrained 4-dimensional fields, as well as with the remaining 4-dimensional gauge symmetry." (G_{Abc} in (4) is
not defined in the paper; it is the compensating local SO(d) rotation.) The left-hand sides of (2)-(3) are the Lie
derivatives along xi_A.

(b) CSDR constraints and rules (pp. 3-4).
- From (3): "the 4-dimensional gauge field, A_mu is completely independent of the coset space coordinates and
  second, the 4-dimensional gauge fields commute with the generators of the subgroup R in G. This means that the
  surviving gauge group, H, is the subgroup of G that commutes with R, that is the centralizer of R in G, i.e.
  H = C_G(R_G)." (p. 3)
- The A_alpha(x, y) = phi_alpha(x, y) "act as intertwining operators connecting induced representations of R
  acting on G and S/R". Rule (ii), eqs. (5)-(6), p. 3:

$$G \supset R_G \times H, \qquad adj\,G = (adj\,R_G, 1) + (1, adj\,H) + \sum (r_i, h_i), \tag{5}$$

$$S \supset R, \qquad adj\,S = adj\,R + \sum s_i. \tag{6}$$

  "Therefore, we conclude that for every pair r_i, s_i, where r_i and s_i are identical irreducible representations
  of R, there remains a scalar (Higgs) multiplet which transforms under the representation h_i of H. All other
  scalar fields vanish." (p. 3)
- Scalar constraint in explicit form, eq. (12), p. 4 (obtained "considering the constraints of the fields,
  (2)-(3)"):

$$f^D_{ai}\phi_D - [\phi_a, \phi_i] = 0, \tag{12}$$

  "where the phi_i are the generators of R_G". Since [R, coset] is in the coset for a reductive coset, f^D_{ai} is
  non-zero only for D = b, so (12) is [phi_i, phi_a] = f^b_{ai} phi_b up to the sign fixed by the antisymmetry
  f^b_{ai} = -f^b_{ia}; the paper writes it with phi_i identified with the embedded R-generators (the W_i).
- Fermion rule (iii), eqs. (7)-(8), pp. 3-4: decompose the representation F of G under R_G x H and the spinor of
  SO(d) under R,

$$G \supset R_G \times H, \qquad F = \sum (r_i, h_i), \tag{7}$$

$$SO(d) \supset R, \qquad \sigma_d = \sum \sigma_j. \tag{8}$$

  "for each pair r_i and sigma_i, where r_i and sigma_i are identical irreducible representations, there exists a
  multiplet, h_i of spinor fields in the 4-dimensional theory." Chirality (p. 4): starting from Dirac fermions one
  cannot obtain chiral 4D fermions; the Weyl condition in even D gives a chiral 4D theory; for D = 2n + 2 with
  fermions in the adjoint one gets two sets of chiral fermions with the same H quantum numbers, the doubling is
  removed by the Majorana condition, and Weyl + Majorana are compatible when D = 4n + 2 (D = 10 is the case of
  interest).

(c) Reduced 4D action, eqs. (9)-(11), p. 4. With the metric (9), "inserting (9) into the initial action and taking
into account the constraints of the fields, we obtain"

$$A = C\int d^4x \left[ -\tfrac14 F^t_{\mu\nu}F^{t\mu\nu} + \tfrac12 (D_\mu\phi_\alpha)^t (D^\mu\phi^\alpha)^t
 + V(\phi) + \tfrac{i}{2}\bar\psi\Gamma^\mu D_\mu\psi - \tfrac{i}{2}\bar\psi\Gamma^a D_a\psi \right], \tag{10}$$

"where D_mu = partial_mu - A_mu and D_a = partial_a - theta_a - phi_a, with theta_a = (1/2) theta_{abc} Sigma^{bc}
the connection of the space and C is the volume of the space." The potential is

$$V(\phi) = -\tfrac14\, g^{ac}g^{bd}\,\mathrm{Tr}\big(f^C_{ab}\phi_C - [\phi_a,\phi_b]\big)
 \big(f^D_{cd}\phi_D - [\phi_c,\phi_d]\big), \tag{11}$$

"where, A = 1, ..., dim S and f's are the structure constants appearing in the commutators of the Lie algebra of S."
In the tutorial's notation F_{ab} = [phi_a, phi_b] - f^c_{ab} phi_c (the sum over C in (11) runs over all of S but
the phi_i, i in R, are fixed by (12) to the embedded R-generators, so the D-sum contributes both f^c_{ab} phi_c and
f^i_{ab} phi_i terms; the latter is the standard constant/mass-term piece for symmetric cosets). V(phi) is "a quartic
polynomial which is invariant under the 4-dimensional gauge group, H" (p. 4). No explicit coset radii are shown;
their role is hidden in g^{ab} and in the normalisation of C. Note the overall sign: (10) has +V(phi) in the
Lagrangian; with anti-hermitian phi and Tr(X X) <= 0 the expression (11) is >= 0, so with L = T - V_phys the
authors' V is minus the physical potential unless Tr is defined with the opposite sign; the paper does not fix
this (see Caveats). The fermion term -(i/2) bar-psi Gamma^a D_a psi contains the Yukawa couplings (via phi_a in D_a)
and the geometric mass term (via theta_a).

(d) Minimisation of the potential (p. 4, eq. (13)). "Then, it follows the determination of the minimum of the
potential and the finding of the remaining gauge symmetry of the vacuum [42-44]. In general, this is a rather
difficult procedure. However, there is a case in which one could obtain the result of the spontaneous breaking of
the gauge group H very easily, whether the following criterion is satisfied. Whenever S has an isomorphic image
S_G in G, then the 4-dimensional gauge group H breaks spontaneously to a subgroup K, where K is the centralizer of
S_G in the gauge group of the initial, higher-dimensional theory, G [4,42-44]. This can be illustrated in the
following scheme,"

$$\begin{array}{ccc} G \supset S_G \times K \\ \cup \quad \cap \\ G \supset R_G \times H \end{array} \tag{13}$$

i.e. the vacuum phi_a extends the embedding R_G subset G to an embedding S_G subset G, and H = C_G(R_G) breaks to
K = C_G(S_G). "In addition, the potential of the resulting 4-dimensional gauge theory is always of spontaneous
symmetry breaking form, when the coset space is symmetric [fn. 1: f^c_{ab} = 0]. A negative result in this case is
that, after the dimensional reduction, the fermions end up being supermassive, as in the Kaluza-Klein theory."

(e) Explicit example. The ONLY example in the CSDR part is the N = 1, D = 10, E8 theory reduced over the
nearly-Kähler non-symmetric coset SU(3)/U(1)xU(1) (p. 5, eqs. (14)-(15)):

$$E_8 \supset E_6 \times SU(3) \supset E_6 \times U(1)_A \times U(1)_B, \tag{14}$$

$$H = C_{E_8}(U(1)_A \times U(1)_B) = E_6 \times U(1)_A \times U(1)_B. \tag{15}$$

The scalars and fermions come from the 248 of E8 decomposed under U(1)_A x U(1)_B; the result is an N = 1, E6 GUT
with U(1)_A, U(1)_B global. "The potential is fully determined after a lengthy calculation and can be found in ref
[15]" (Manousselis-Zoupanos, JHEP 0411 (2004) 025); subtracting the F- and D-terms gives scalar masses and trilinear
terms identified with the scalar soft SUSY-breaking sector; the gaugino mass "receives a contribution from the
torsion". Further breaking E6 -> SU(3)^3 needs the Wilson flux mechanism, ref [12]. The explicit potential, the
SU(2)/U(1) = S^2 example, the Manton SU(3) -> SU(2)xU(1) model and a value of sin^2 theta_W are NOT given in this
paper (the Manton model appears only as ref [7] in the introduction).

(f) Symmetric vs non-symmetric cosets. Definition: footnote 1, p. 4, symmetric iff f^c_{ab} = 0. Consequences
stated: (p. 4) for symmetric cosets V is always of SSB form but fermions are supermassive; (p. 2, intro) the 4D
theories obtained from a 10D N = 1 SYM over NON-symmetric cosets "contain supersymmetry breaking terms [14],[15],
contrary to CY spaces"; the 6D homogeneous nearly-Kähler manifolds are the non-symmetric cosets G2/SU(3),
Sp(4)/(SU(2)xU(1))_{non-max}, SU(3)/U(1)xU(1) and the group manifold SU(2)xSU(2) (p. 2, ref [13]); these admit a
spinor covariantly constant w.r.t. a connection with torsion (p. 2). No formula for the cubic f^c_{ab} phi^a phi^b
phi^c term is written out; it is implicit in (11).

Fuzzy analogue for comparison only (Section 5, p. 10): the fuzzy CSDR constraints read
L_b phi = delta_{W_b} phi = W_b phi (55), L_b A = delta_{W_b} A = -D W_b (56); with the covariant coordinates
phi_a = X_a + A_a (46) and omega_a = X_a - W_a (58) they become [omega_b, A_mu] = 0 (59), C_{bde} phi^e =
[omega_b, phi_d] (60), with consistency condition [omega_a, omega_b] = C^c_{ab} omega_c (61). (59)-(60) are the
discrete versions of the A_mu condition and of (12), and (61) states that the compensating omega_a (the W's)
furnish a representation of the isometry algebra, exactly the "W_i generate R_G" condition. The fuzzy field strength
F_{ab} = [phi_a, phi_b] - C^c_{ab} phi_c (48) has the same form as the continuous F_{ab} in (11).

# Attributable statements

1. (Sec. 2, p. 2-3) The CSDR idea: instead of demanding the fields be independent of the extra coordinates
   (trivial reduction), one allows a non-trivial dependence such that a transformation by an element of the isometry
   group S of the coset B = S/R is a gauge transformation (symmetric fields); gauge invariance of the Lagrangian then
   renders it independent of the extra coordinates. Pioneers: Forgacs-Manton [3] and Kubyshin et al. [5].
2. (Sec. 2, p. 3, eqs. (2)-(4)) The constraints on scalar, vector and spinor fields follow from the definition of
   symmetric fields: the S-transformation of each field equals a gauge transformation with parameter W_A(y).
3. (Sec. 2, p. 3) Solving (3): A_mu is independent of the coset coordinates and commutes with the generators of R in
   G; the 4D gauge group is H = C_G(R_G).
4. (Sec. 2, p. 3, eqs. (5)-(6)) The surviving Higgs multiplets are h_i of H for every pair r_i (from adj G under
   R_G x H) and s_i (from adj S under R) of identical R-irreps.
5. (Sec. 2, pp. 3-4, eqs. (7)-(8)) The analogous fermion rule with the spinor of SO(d) under R; chiral 4D fermions
   need Weyl fermions in even D, and Weyl + Majorana are compatible in D = 4n + 2.
6. (Sec. 2, p. 4, eqs. (10)-(11)) The reduced 4D action has YM kinetic terms, scalar kinetic terms, Yukawa terms from
   the internal Dirac operator and the quartic potential V = -(1/4) g^{ac} g^{bd} Tr(f^C_{ab} phi_C - [phi_a, phi_b])
   (f^D_{cd} phi_D - [phi_c, phi_d]).
7. (Sec. 2, p. 4, eq. (12)) The scalars obey f^D_{ai} phi_D - [phi_a, phi_i] = 0 with phi_i the generators of R_G;
   this filters out some fields and leaves the genuine Higgs fields.
8. (Sec. 2, p. 4, eq. (13)) Vacuum criterion: whenever S has an isomorphic image S_G in G, H breaks spontaneously to
   K = C_G(S_G); for symmetric cosets V is always of SSB form but the fermions are supermassive.
9. (Sec. 2, p. 5, eqs. (14)-(15)) The E8 over SU(3)/U(1)xU(1) reduction gives H = E6 x U(1)_A x U(1)_B with the U(1)s
   global, an N = 1 E6 GUT whose soft SUSY-breaking sector (scalar masses, trilinears, and a torsion-dependent
   gaugino mass) arises from the CSDR without extra assumptions; Wilson flux then breaks E6 to SU(3)^3.
10. (Sec. 2, p. 2) Non-symmetric nearly-Kähler cosets (G2/SU(3), Sp(4)/(SU(2)xU(1))_{non-max}, SU(3)/U(1)xU(1),
    SU(2)xSU(2)) lead to 4D theories with soft supersymmetry breaking terms, unlike CY compactifications.

# Caveats

- Consistency of the truncation: the paper asserts (p. 3) that the symmetric-field ansatz plus gauge invariance
  makes the Lagrangian independent of y, and that inserting (9) and the constraints into (1) gives (10); it does not
  discuss whether the reduced equations of motion coincide with the restriction of the higher-dimensional ones
  (consistent truncation), nor the dependence on whether S/R is taken with the metric induced from a bi-invariant
  metric on S. Treat (10)-(11) as a statement of the result, citing [3,4,5] for derivations.
- Sign of V and trace convention: (10) has +V(phi) inside the Lagrangian while (11) carries an overall -1/4 and a
  trace over anti-hermitian matrices; which sign of Tr makes V positive is not specified. The tutorial must fix this
  independently (Kapetanakis-Zoupanos Phys. Rep. 219 (1992), ref [4], uses the same expressions).
- Index sloppiness: the same letters a, b, c serve as flat coset indices in g^{ab}, as coset generator labels in
  f^c_{ab}, and the sum index C, D in (11) runs over all of S (A = 1, ..., dim S), so the mixed terms f^i_{ab} phi_i
  with phi_i = embedded R-generators are included in (11) implicitly. Also alpha in (3) is curved while a in (10)-(12)
  is flat, and both are called "phi_alpha"/"phi_a" without an explicit vielbein.
- Equation (12) is written with phi_i "the generators of R_G"; the paper does not say explicitly that these are
  determined by the W_A of (2)-(4) (the W_i restricted to R), nor that the W_a for coset generators can be set to
  zero by a gauge choice. G_{Abc} in (4) is undefined.
- Coset radii: none appear; the coset metric g^{ab} is generic, and the volume C multiplies the whole 4D action.
  Anything the tutorial says about the dependence of V on radii R_a cannot be attributed to this paper.
- Mass terms: for symmetric cosets the fermions "end up being supermassive, as in the Kaluza-Klein theory" (p. 4);
  the paper gives no formula for the mass (it comes from theta_a in D_a). The scalar mass terms in V (from the
  f^i_{ab} phi_i cross-terms) are not isolated.
- Torsion: mentioned only qualitatively: SU(3)-structure manifolds have a spinor constant w.r.t. a connection with
  torsion (p. 2), and in the E8 model "the gaugino obtains a mass, which receives a contribution from the torsion"
  (p. 5). The torsion-modified connection is not written.
- Non-symmetric cosets: the only statement is that non-symmetric (nearly-Kähler) cosets lead to SUSY-breaking terms
  and that the symmetric-case SSB-form guarantee does not extend; the cubic f^c_{ab} term is not displayed.
- No explicit potential, no SU(2)/U(1) or Manton SU(3) example, no sin^2 theta_W: the E8 potential is deferred to
  ref [15] and the SU(3)^3 breaking to ref [12]. Do not cite this paper for those.
- Sections 3-6 (fuzzy sphere, fuzzy CSDR, orbifolds, trinification) are outside the tutorial's scope; only the
  structural parallel (59)-(61) and (48) with (11)-(12) is worth a remark.
