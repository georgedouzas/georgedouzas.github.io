# Theory of Spinors and Its Application in Physics and Mechanics

Vladimir A. Zhelnorovich, Springer, 2019. Local copy: `references/textbooks/Theory of Spinors and Its Application in Physics and Mechanics.pdf` (git-ignored). Textbook (design D32).

## Chapters it serves

Mapped from the book's table of contents (chapter titles and page numbers of the PDF). The sections, equations
and problem numbers actually used are added below when the chapter that uses them is written or revised.

| Chapter of this book | Sections of the textbook | Use |
|---|---|---|
| spinors and the Dirac operator (new Part I chapter, D31) | Ch. 1 (spinors in n-dimensional complex Euclidean spaces: Clifford algebras, gamma matrices in any dimension), Ch. 2 (spinors in Riemannian spaces, spin connection), Ch. 3-4 (spinors in four and three dimensions, tensor representations) | content and examples |
| the Dirac equation (new Part II chapter, D31) | Ch. 5 (relativistically invariant spinor equations), Ch. 6 (exact solutions, including the Einstein-Dirac equations in homogeneous spaces) | content and examples, the homogeneous-space solutions for the coset reductions |
| part-ii-10-supersymmetry-supergravity | Ch. 1 (Majorana and charge-conjugation matrices in any dimension) | examples |

## Sections, equations and problems used

Page offset: PDF page = printed page + 14. The book has no numbered problems; examples and exercises adapted from it cite
sections and equations.

part-i-13-spinors-dirac-operator (Spinors and the Dirac operator):

- Sec. 1.1, items 1-6, eqs. (1.1)-(1.12) (products gamma_A, squares (1.4), vanishing traces (1.6)-(1.7), linear independence and
  the 2^{2 nu} basis (1.8), expansion (1.10)-(1.12)): prp-clifford-basis, exm-spd-squares-count, exr-spd-trace-zero,
  exr-spd-expand-matrix.
- Sec. 1.1, item 8, eq. (1.19) (Pauli identity): exm-spd-basis-two.
- Sec. 1.1, item 9, eq. (1.21) (gamma_{2 nu + 1} = i^nu gamma_1 ... gamma_{2 nu}): def-chirality-matrix.
- Sec. 1.1, items 10-11, eqs. (1.26)-(1.35) (commutant, Pauli's theorem, averaging formula (1.29)): thm-pauli-uniqueness,
  exr-spd-commutant, exr-spd-pauli-average, exm-spd-odd-inequivalent; item 10 (centre of the algebra in odd and even
  dimension): exr-spd-centre, and the solution of exr-spd-commutant.
- Sec. 1.1, eqs. (1.36)-(1.43) (existence by the recursion adding two dimensions): exr-spd-recursion, def-gamma-matrices-general.
- Sec. 1.1, eqs. (1.44)-(1.56) (matrix E with gamma^T = -E gamma E^{-1}, transformation (1.48), symmetry (1.53)):
  exm-spd-transpose-sets, def-charge-conjugation-matrix, exr-spd-charge-basis.
- Sec. 1.2.1, eqs. (1.57)-(1.69) (spinor representation l gamma = S^{-1} gamma S, normalisation (1.60), +-S, generators (1.68),
  infinitesimal S (1.69)): def-spin-group, prp-spin-lie-algebra.
- Sec. 1.5 (semispinors): def-weyl-representation.
- Sec. 1.6.2, eq. (1.120) (gamma matrices of signature with q imaginary factors): def-clifford-algebra.
- Sec. 1.6.2-1.6.3, eqs. (1.127)-(1.142) (matrix beta, Sigma = E^{-1} beta^T, Sigma Sigma-bar = +-1, real and imaginary
  representations): exm-spd-transpose-sets, def-majorana-condition, exm-spd-majorana-four, exr-spd-imaginary-rep.
- Secs. 1.2 and 3.1, cited in the introduction of the part on spin groups (before def-spin-group); Secs. 2.2-2.3, cited in the
  introduction of the part on spin structures (before def-spin-structure).
- Sec. 2.2.1, eqs. (2.26)-(2.35) (Ricci rotation coefficients): def-spin-connection.
- Sec. 2.3, eqs. (2.50)-(2.69) (spinor connection Gamma_i = (1/4) Delta_{i,ab} gamma^{ab} (2.56), transformation (2.61), (2.65),
  covariant derivative (2.68)): def-spin-connection, exm-spd-frame-change, prp-spinor-curvature.
- Sec. 3.1.3, eqs. (3.30)-(3.45) (spinor representations of the Lorentz group; rotation S = cos(phi/2) + gamma^{ab} sin(phi/2)
  (3.45), boosts): exm-spd-spin-two, exm-spd-spin-lorentz, prp-spin-lie-algebra (eq. (3.44)).
- Sec. 4.1.3, eqs. (4.15)-(4.18) and the formula S = exp(-(i/2) phi n.sigma): exm-spd-spin-three.
- Sec. 6.1, eqs. (6.1)-(6.2) (Ricci rotation coefficients and Ricci tensor in orthonormal frames, used with constant
  coefficients on group manifolds): def-left-invariant-dirac and the introduction of sec-dirac-compact-spaces.


part-ii-05-dirac-equation (The Dirac equation):

- Sec. 3.2.1, eqs. (3.57)-(3.59) (scalar, vector, tensor, pseudovector, pseudoscalar of a spinor): def-dq-bilinear-covariant.
- Sec. 3.2.1, eqs. (3.60) (Pauli-Fierz identities, converted from signature (+,+,+,-)) and the positivity of j^4 after (3.62):
  exm-dq-fierz, exr-dq-fierz-check, prp-dq-current-conservation.
- Sec. 5.1, eqs. (5.1)-(5.6) (invariance condition for G^j d_j psi + F psi = 0) and (5.11)-(5.13) (infinitesimal condition, order
  four of the matrices, generators up to a multiple of 1): thm-dq-covariance, exr-dq-infinitesimal, prp-dq-matrix-size, exr-dq-eight.
- Sec. 5.2, eq. (5.17) (general linear spinor equation with tensor coefficients): exm-dq-general-equation; eq. (5.19) (Dirac
  equation in his conventions): exr-dq-zhelnorovich-form.
- Sec. 5.7, eqs. (5.79)-(5.81) (Weyl equations for the two semispinors): def-dq-weyl-equation; Sec. 5.7 is also cited in the
  introduction of the part on Weyl and Majorana fermions (before def-dq-charge-conjugate).
- Sec. 5.11.2 (spinor equations in spherical coordinates): prp-dq-radial-equations.
- Ch. 6 (exact solutions of nonlinear spinor equations, Einstein-Dirac in homogeneous spaces): read in outline, not used in this
  chapter (it concerns nonlinear and gravitational equations).

part-vi-02-compactifications (Fermions on compact spaces, task 12.7):

- Sec. 6.1, eqs. (6.3)-(6.7) (Einstein-Dirac equations in signature (+,+,+,-), energy-momentum tensor of the spinor, trace
  R = kappa m rho cos eta): exm-cff-einstein-dirac.
- Sec. 6.2, eqs. (6.14)-(6.22) (general solution in the homogeneous Bianchi I space: synchronism (6.16), integrals (6.17),
  eq. (6.18), solution (6.21) with constants (6.22), oscillation with frequency 2m): exm-cff-einstein-dirac,
  exr-cff-bianchi. Page offset for Ch. 6: PDF page = printed page + 11.

part-ii-10-supersymmetry-supergravity (textbook pass):

- Sec. 1.1, eqs. (1.44)-(1.55) (matrix E with gamma^T = -E gamma E^{-1}, E^T = (-1)^{nu(nu+1)/2} E, symmetry of E gamma_{i1...ik}):
  exm-ssg-c-general, exr-ssg-zhelnorovich-e (Sage).
- Secs. 1.6.2-1.6.3, eqs. (1.127)-(1.142) (complex conjugation matrix, its square +-1, real and imaginary representations): exr-ssg-majorana-bb
  (Sage), restated as the consistency B^*B = 1 of a Majorana condition in signature (+, -, ..., -).
