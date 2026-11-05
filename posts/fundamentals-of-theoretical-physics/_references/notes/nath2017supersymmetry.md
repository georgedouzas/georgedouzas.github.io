# Supersymmetry, Supergravity, and Unification

Pran Nath, Cambridge University Press (Cambridge Monographs on Mathematical Physics), 2017. Local copy: `references/textbooks/Nath - 2016 - Supersymmetry, Supergravity, and Unification.pdf` (git-ignored). Textbook (design D32).

## Chapters it serves

Mapped from the book's table of contents (chapter titles and page numbers of the PDF). The sections, equations
and problem numbers actually used are added below when the chapter that uses them is written or revised.

| Chapter of this book | Sections of the textbook | Use |
|---|---|---|
| part-ii-10-supersymmetry-supergravity | Ch. 8 (supersymmetry, p. 164ff), Ch. 11-12 (N = 1 supergravity and its couplings, p. 272ff), Ch. 18 (extended supergravities and supergravities from superstrings, p. 429ff) | content and examples |
| part-ii-06-gauge-theories | Ch. 3-4 (non-abelian gauge theory, symmetry breaking, p. 77-103), Ch. 6 (anomalies, p. 115ff) | examples |
| renormalisation (new Part II chapter, D31) | Ch. 7 (effective Lagrangians, p. 141ff), Ch. 9 (grand unification and running couplings, p. 209ff) | content for effective field theories and running couplings |

## Sections, equations and problems used

### part-ii-08-renormalisation

- Sec. 7.1, eqs. 7.10-7.19 and 7.29 (effective Lagrangians from single-particle saturation, Weinberg sum rule):
  exm-ren-match-hadrons, sec-effective-field-theories introduction.
- Eqs. 9.33-9.40 (couplings at M_Z, one-loop beta functions with C_2 and T(R), SM coefficients (41/10, -19/6, -7)):
  prp-ren-beta-functions, exm-ren-bf-sm, exr-ren-bf-sm, fig-ren-unification, exr-ren-lp-unification, exr-ren-thr-nath.
- Sec. 9.3, eqs. 9.85-9.92 (MSSM coefficients (33/5, 1, -3), step-function thresholds, the relation for M_H3):
  exm-ren-bf-sm, exm-ren-lp-unification, def-ren-threshold-correction, fig-ren-thresholds, exm-ren-thr-nath, exr-ren-thr-nath.

### part-ii-10-supersymmetry-supergravity (textbook pass)

Page offset: PDF page = printed page + 25. Ch. 8 uses the metric diag(-1, 1, 1, 1); every formula is converted to (+, -, -, -)
(sign of the Box terms, of the epsilon terms and of the scalar kinetic terms noted at the point of use).

- Sec. 8.1, eqs. 8.4-8.15 (Coleman-Mandula, Casimirs P^2 and W^2): exm-ssg-coleman-mandula.
- Problem 8.2 (products of three and four sigma matrices): exr-ssg-sigma-triple (Sage). Problem 8.4 (decomposition of
  A_{alpha beta alphadot betadot}): exr-ssg-bispinor-decomposition.
- Secs. 8.4-8.5, eqs. 8.59-8.82 and 8.102-8.106 (chiral and antichiral superfields, expansion in x): def-ssg-chiral-superfield,
  exm-ssg-superfield-x-expansion, exm-ssg-antichiral, exr-ssg-theta-fierz (Sage), exr-ssg-antichiral-expansion.
- Secs. 8.6-8.8, eqs. 8.108-8.111, 8.127-8.128, 8.135, 8.142-8.145 (products of chiral superfields, F-term of W, kinetic term,
  single-field model): prp-ssg-chiral-products, exm-ssg-product-mass, exm-ssg-product-cubic, exm-ssg-kahler-term,
  exr-ssg-chiral-products (Sage), exr-ssg-single-field (also eqs. 8.250-8.253).
- Secs. 8.9-8.10, eqs. 8.147-8.170, 8.185, Problem 8.9 (vector superfield, Wess-Zumino gauge, W_alpha): exr-ssg-vector-superfield,
  exr-ssg-field-strength-superfield.
- Sec. 8.18, eqs. 8.300-8.307, Problem 8.14 (spurion soft terms, A0 = 2B0): exr-ssg-spurion-soft (Sage).
- Sec. 8.19, eqs. 8.309-8.318 (R-symmetry, R-charges of components, gaugino masses): def-ssg-r-symmetry, exm-ssg-r-nath,
  exm-ssg-r-oraifeartaigh, exm-ssg-r-gaugino, exr-ssg-r-assign (Sage), exr-ssg-r-components.
- Ch. 10, Table 10.1 and eq. 10.16 (two Higgs doublets, R-parity-violating terms): exr-ssg-higgsino-anomaly, exr-ssg-r-parity.
- Ch. 10, eqs. 10.43-10.49 (chargino matrix and masses), eqs. 10.51-10.53 (neutralino matrix): exr-ssg-chargino-masses (Sage),
  exm-ssg-neutralino.
- Ch. 12, eqs. 12.5-12.16 (W, K, G = K + log|W|^2, scalar potential 12.12-12.14, Kahler transformations): def-ssg-kahler-potential,
  exm-ssg-kahler-canonical, exm-ssg-kahler-shift, exr-ssg-kahler-g-invariance, prp-ssg-sugra-potential, exm-ssg-sugra-ads,
  exm-ssg-polonyi, exm-ssg-sugra-global-limit, exr-ssg-polonyi (Sage), exr-ssg-sugra-g-form.
- Problem 12.1 (no-scale potential): exr-ssg-no-scale (Sage); the field space of K = -n log(T + Tbar): exm-ssg-kahler-no-scale,
  exr-ssg-kahler-curvature (Sage), exr-ssg-kahler-two-moduli.
- Sec. 18.1.1, eqs. 18.23-18.24 and 18.29, Problem 18.1 (reduction of eleven-dimensional supergravity to IIA, Einstein frame):
  exr-ssg-iia-reduction.
- Secs. 18.2.3-18.2.5 (heterotic multiplets, 496, Green-Schwarz): exr-ssg-heterotic-count.
- Sec. 18.4, eqs. 18.90-18.92 (gravitino decay and the nucleosynthesis bound): exr-ssg-gravitino-lifetime; Sec. 18.4 is also
  cited in exm-ssg-kahler-no-scale for moduli from compactification.
- Ch. 10 (MSSM, not in the mapping table) is used as listed above; eqs. 8.165-8.170 and 8.185 are cited in the solution of
  exr-ssg-field-strength-superfield, Ch. 10 with Table 10.1 before def-f-term-potential, eqs. 10.44 and 10.49 in the
  solution of exr-ssg-chargino-masses, Secs. 18.2.3 and 18.2.5 in the solution of exr-ssg-heterotic-count, eqs. 18.24 and
  18.29 in the solution of exr-ssg-iia-reduction.
- Not used (task 8 check): Ch. 11 (N = 1 supergravity with auxiliary fields, tensor calculus). The chapter builds the
  gravitino, the supergravity multiplet and the eleven-dimensional theory from Tanii (hep-th/9802138), and needs from N = 1 supergravity only the matter couplings, which are taken from Ch. 12; the
  off-shell tensor calculus of Ch. 11 is beyond the chapter's statements. Sec. 18.3 (four-dimensional string models) is
  not used either, being phenomenology outside the chapter's line.

### part-ii-06-gauge-theories (textbook pass)

- Sec. 3.1, eqs. 3.1-3.3 (covariant derivative from parallel transport): exm-gth-parallel-transport.
- Problem 3.2 (self-duality bound of the Euclidean action, eq. 3.150): exr-gth-self-dual-bound.
- Problem 3.3 and eq. 6.119 (theta term as a total divergence, Chern-Simons form): exr-gth-theta-total-derivative (Sage).
- Problem 3.6, eq. 3.155 (Yang-Mills energy in the temporal gauge): exr-gth-ym-energy.
- Sec. 4.1, eqs. 4.13-4.19 (Goldstone-vector mixing, change of variables, degree-of-freedom count): exr-gth-cartesian-mixing.
- Problem 4.2 (gauge transformation in Cartesian fields): exr-gth-cartesian-gauge.
- Sec. 4.2, eqs. 4.26-4.33 (Weinberg's existence of the unitary gauge, mass matrix, R_xi gauges): exm-gth-weinberg-unitary.
- Secs. 6.1 and 6.9-6.10, eqs. 6.123-6.126, 6.136-6.139 (anomaly constraints, safe groups, Witten's anomaly, Standard Model
  checks): def-gth-anomaly-coefficient, exm-gth-anomaly-u1, exm-gth-anomaly-su2, exm-gth-anomaly-su3, exr-gth-anomaly-su3 (Sage),
  exr-gth-anomaly-u1-charges (Sage), exr-gth-anomaly-mixed, prp-gth-anomaly-conjugate, exm-gth-anomaly-su5,
  exm-gth-anomaly-safe-groups, exm-gth-witten-anomaly, exr-gth-anomaly-tensor, exr-gth-anomaly-colour.
- Eq. 6.123 and Problem 6.7 (anomaly of antisymmetric tensors of SU(N), anomaly-free combinations): exr-gth-anomaly-antisymmetric (Sage).
- Error noted: eq. 6.125 lists all SO(2N), N >= 2, among the groups with only real or pseudo-real representations; the spinors of
  SO(4N + 2) are complex (they are covered by eq. 6.126), and SO(6) = SU(4) is not anomaly-free, so eq. 6.126 holds for N >= 2.
