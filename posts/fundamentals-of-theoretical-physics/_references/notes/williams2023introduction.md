# Introduction to Quantum Field Theory

Anthony G. Williams, Cambridge University Press, 2023. Local copy: `references/textbooks/Williams - 2023 - Introduction to Quantum Field Theory.pdf` (git-ignored). Textbook (design D32), from the author's Zotero library
for change kk-csdr-physics-foundations (design E5).

## Chapters it serves

Mapped from the book's table of contents. The sections, equations and problem numbers actually used are added below
when the chapter that uses them is written or revised.

| Chapter of this book | Sections of the textbook | Use |
|---|---|---|
| part-ii-01-classical-mechanics | Sec. 2.9 (constrained Hamiltonian systems: Dirac-Bergmann algorithm, gauge fixing, Dirac bracket, quantisation) | content (task 5.6) |
| part-ii-07-quantum-field-theory | Ch. 6 (fermions, photons, massive vector bosons), Ch. 7 (spectrum, S-matrix, cross sections and decay rates, Feynman rules) | content (task 5.5) |
| part-ii-06-gauge-theories | Ch. 9 (non-abelian gauge theories: gauge fixing, Faddeev-Popov, BRST; QCD; anomalies; the Standard Model) | content (tasks 5.4, 3.2) |
| part-ii-06-gauge-theories | Sec. 9.1.2 (Wilson lines and loops), Sec. 9.2.5 (lattice QCD: links, plaquettes, Wilson action, Haar measure, Monte Carlo) | content (task 3.1) |

## Sections, equations and problems used

### part-ii-06-gauge-theories (physics foundations, task 3.1: lattice gauge theory)

Williams writes the link from $x$ to $x + a\hat\mu$ with Hermitian generators, $U_\mu(x) = e^{iag_0A_\mu(x)}$; the chapter uses
anti-Hermitian generators and the transporter from $x + a\hat\mu$ to $x$, so the formulas are translated.

- Sec. 9.1.2, eqs. (9.1.13)-(9.1.19) (Wilson lines, path ordering, Wilson loops, non-abelian Stokes theorem):
  def-lgt-wilson-loop, exm-lgt-wl-continuum, exr-lgt-plaq-two-squares.
- Sec. 9.2.5, eqs. (9.2.54)-(9.2.57) (links, counting of link variables, Haar measure and its properties (i)-(iv)):
  def-lgt-link-variable, exm-lgt-link-count, exm-lgt-wa-link-zero, def-lgt-wilson-action.
- Eqs. (9.2.58)-(9.2.63) (plaquette, lattice field strength by Baker-Campbell-Hausdorff, Wilson gauge action with
  $\beta = 6/g_0^2$): def-lgt-plaquette, prp-lgt-continuum-limit, exm-lgt-wa-su3.
- Eqs. (9.2.64)-(9.2.65) (gauge-invariant hopping terms, covariant difference): exm-lgt-link-hopping.
- Eqs. (9.2.66)-(9.2.68) (fermion determinant, quenched approximation, hybrid Monte Carlo): exm-lgt-hb-fermions.
- Sec. 9.2.5, gauge fixing on the lattice (Landau gauge, Gribov copies): exm-lgt-el-gauge-fixed.
- Sec. 9.2.5, chiral symmetry breaking studied in lattice QCD: exm-lgt-el-chiral.
- Errors found (Sec. 9.2.5): in the proof of eq. (9.2.60), "$A + B + C + D = ia^2g_0(\Delta_\mu A_\mu - \Delta_\nu A_\mu)$" should read
  $\Delta_\mu A_\nu - \Delta_\nu A_\mu$, and the exponent "$-ig_0[A_\mu, A_\mu]$" should be $-ig_0[A_\mu, A_\nu]$; on p. 710 the naive
  lattice fermion has $\tilde k_\mu = \frac1a\sin(k_\mu a)$, not $\frac12\sin(k_\mu a)$.
