# Identity

- arXiv: hep-ph/0503177 (v2); Journal of Physics: Conference Series 18 (2005) 224-269.
- Title: An Introduction to Extra Dimensions. Author: Abdel Pérez-Lorenzana. Lectures at the XI Mexican School of
  Particles and Fields (2004). 39 PDF pages.
- Read: Sec. III.A (bulk fermions), the supersymmetry paragraph of Sec. III.C, Secs. IV.C (orbifold breaking) and IV.D
  (Scherk-Schwarz mechanism), Sec. V.1 (localising wave functions on the brane).

# Structure

I introduction; II flat compact extra dimensions (ADD); III model building: A bulk fermions, B bulk vectors, C short and
large extra dimensions (UED, supersymmetry); IV symmetry breaking with extra dimensions: A spontaneous breaking in the
bulk, B shining, C orbifold breaking, D Scherk-Schwarz; V lepton and baryon number in low-gravity models (1: localising
wave functions); VI warped extra dimensions.

# Conventions and mapping to the tutorial

Signature (+,-,-,-,-) with {Gamma^M, Gamma^N} = 2 eta^{MN}; Gamma^mu = gamma^mu in the chiral basis, Gamma^4 = i gamma_5
(eq. 35), the same choice as eq-spd-product-gammas of the tutorial. Psi = (psi_R, psi_L) (eq. 36). The circle has radius
R (fields expanded in cos(ny/R), sin(ny/R)). In Sec. V.1 the action has + <Phi> psibar psi (eq. 98), so his Phi is minus
the tutorial's bulk mass M; with that sign change the right-handed zero mode is exp(-M|y|), as in eq-kkf-orbifold-zero-mode.

# Equations the tutorial uses

- Eqs. (35)-(36): five-dimensional gamma matrices, no chirality in five dimensions, a 5D fermion is a 4D Dirac spinor.
- Eqs. (38)-(39): mode expansion on the circle, the term psibar gamma_5 d_y psi becomes Dirac mass terms n/R pairing even
  and odd modes.
- Eq. (40) and the following paragraph: 5D Dirac mass gives masses to all modes; it is odd under y -> -y with
  Psi -> +-gamma_5 Psi and is forbidden on the orbifold; the orbifold leaves one chirality at the zero-mode level.
- Sec. III.C, supersymmetry paragraph: 5D N = 2 multiplets; hypermultiplets are chiral + antichiral N = 1 multiplets, the
  chiral ones even under Z_2 with massless states.
- Sec. IV.C, eqs. (68)-(72): orbifold parities of a doublet, D_mu even and D_5 odd, A_mu -> P A_mu P^-1, A_5 -> -P A_5 P^-1.
- Sec. IV.D, eqs. (74)-(79): Scherk-Schwarz twist phi(y + 2 pi R) = exp(2 pi i beta) phi(y), modes exp(i(n + beta)y/R),
  masses (n + beta)/R, breaking of global and (Hosotani) gauge symmetries, mention of supersymmetry breaking.
- Sec. V.1, eqs. (98)-(103): fermion zero modes in a kink background, profiles exp(-+ int Phi), localisation, several
  fermions at different positions.

# Sections and equations used (part-iii-02-gauge-tower, sec-kk-fermions)

- Eqs. (35)-(38): def-kkf-five-dirac, exm-kkf-flat-five.
- Eqs. (38)-(40): prp-kkf-vectorlike, exm-kkf-bulk-mass, exr-kkf-reflection.
- Eq. (40) and following paragraph, Sec. IV.C eqs. (69)-(72), Sec. V.1 eqs. (98)-(101): prp-kkf-orbifold-zero-mode,
  exm-kkf-orbifold-gauge; eq. (100): exr-kkf-massive-orbifold; eqs. (98)-(103): exm-kkf-localisation, exr-kkf-overlap.
- Sec. IV.D eqs. (75)-(77): def-kkf-fermion-tower, exm-kkf-scherk-schwarz, exr-kkf-wilson-line.
- Sec. III.C (supersymmetry paragraph): exr-kkf-int-counting.

# Caveats

- The lectures are an effective-field-theory review; the statements on supersymmetry breaking by twists are qualitative.
- No spin connection or Kaluza-Klein gauge field enters the fermion sections; the Pauli term of eq-kkf-dirac is derived in
  the tutorial (exr-kkf-cartan, exr-kkf-gamma-contraction), not taken from here.
