# Applied Computational Physics

Joseph F. Boudreau and Eric S. Swanson, Oxford University Press, 2018. Local copy: `references/textbooks/Boudreau and Swanson - 2017 - Applied Computational Physics.pdf` (git-ignored). Textbook (design D32).

## Chapters it serves

Mapped from the book's table of contents (chapter titles and page numbers of the PDF). The sections, equations
and problem numbers actually used are added below when the chapter that uses them is written or revised.

| Chapter of this book | Sections of the textbook | Use |
|---|---|---|
| part-ii-02-thermodynamics-statistical-mechanics | Ch. 7 (Monte Carlo, p. 219ff), Ch. 8 (percolation and universality, p. 263ff), Ch. 20 (classical spin systems, p. 724ff) | Sage exercises |
| part-i-11-lie-groups-algebras | Ch. 14 (rotations and Lorentz transformations, p. 478ff) | examples |
| part-ii-08b-special-relativity | Ch. 14 (Secs. 14.3-14.5, Exercises 14.4, 14.6) | content and exercises |
| part-ii-07-quantum-field-theory | Ch. 24 (quantum field theory on the lattice, p. 867ff) | examples and exercises |
| part-ii-08-renormalisation | Ch. 8 (Sec. 8.4, coarse graining), Ch. 24 (lattice regularisation, fermions) | content, examples and exercises (row added by the task 8 check; the uses were already listed below) |
| part-ii-06-gauge-theories | Ch. 24 (Secs. 24.3-24.5: Z2, compact U(1) and SU(2) lattice gauge theory; Exercises 24.2-24.3, 24.6-24.11, 24.14-24.17, 24.19) | content, examples and exercises (physics foundations, task 3.1) |

## Sections, equations and problems used

### part-ii-08-renormalisation

- Sec. 8.4.1, eqs. 8.49-8.53 (coarse graining, triangular percolation, nu = 1.355): exm-ren-bs-percolation, exr-ren-bs-percolation,
  exm-ren-fp-percolation.
- Eqs. 8.54-8.59 (theory space, renormalisation map, fixed points, linearisation): def-ren-rg-map, def-ren-fixed-point.
- Exercise 8.13 (decimation of the Ising chain): exm-ren-bs-ising-1d, exr-ren-bs-1d-ising, exm-ren-fp-chain.
- Exercise 8.14 (approximate map for the square lattice): exm-ren-bs-ising-2d, exr-ren-bs-2d-ising, exr-ren-x-nu-2d.
- Exercise 8.15 (central limit theorem as a Gaussian fixed point): exr-ren-bs-clt.
- Sec. 24.2, eqs. 24.5-24.7 (the lattice as regulator), Sec. 24.4.4 (continuum limit), Sec. 24.6 (fermions):
  def-ren-lattice-regularisation, exm-ren-lattice-doublers. Exercise 24.1 (Wilson fermions): exr-ren-wilson-fermion.

### part-ii-02-thermodynamics-statistical-mechanics (task 12.8)

- Sec. 7.5.1 (Metropolis-Hastings): exm-tsb-metropolis-two-state.
- Ch. 7, Exercise 13 and Ch. 20, Exercise 17 (heat bath and Glauber updates obey detailed balance): exr-tsb-detailed-balance.
- Ch. 7, Exercise 16 (three-state Markov chain): exr-tsb-markov (Sage).
- Secs. 8.1, 8.3 (site percolation, percolation probability, critical exponents): exm-tsb-percolation.
- Ch. 8, Exercise 4 (percolation on a Cayley tree, p_c, beta, gamma): exm-tsb-cayley-threshold, exr-tsb-cayley-exponents.
- Ch. 8, Exercise 5 (one-dimensional percolation, nu = 1): exr-tsb-chain-percolation.
- Ch. 20, Exercise 1 (delta = 15 for the two-dimensional Ising model): exm-tsb-delta-fifteen.
- Sec. 20.3 and Ch. 20, Exercise 10 (Potts model, first-order transitions for q > 4): exm-tsb-potts.
- Ch. 20, Exercise 9 (Potts correlations, here on a chain; the book's vector normalisation differs by 1/q): exr-tsb-potts-chain (Sage).
- Sec. 20.4 and Ch. 20, Exercise 11 (XY chain, Z = (2 pi)^L I_0^(L-1)): exm-tsb-xy-pair, exr-tsb-xy-chain (Sage).
- Ch. 20, Exercise 13 (Baxter-Wu ground states): exr-tsb-baxter-wu.

### part-ii-08b-special-relativity (task 12.9)

- Sec. 14.3, eq. (14.4) (representations $(j_1, j_2)$ from two spin matrices, $R = \exp(-i(\vec n \pm i\vec\eta)\cdot\vec J)$): def-sr-j1j2, prp-sr-complexification.
- Sec. 14.5.2, eq. (14.9) (four-momentum of a Weyl spinor $s^\dagger\sigma^\mu s$): exm-sr-vector-from-spinors.
- Sec. 14.5.4, eqs. (14.14)-(14.16) (antisymmetric tensors as two complex three-vectors $\vec E \pm i\vec B$): exm-sr-field-pair, exr-sr-field-complex (our
  assignment of $(1, 0)$ to $\vec E - i\vec B$ follows from the chapter's sign conventions and is checked in Sage).
- Exercise 14.4 (Lorentz matrix acting on vectors and field tensors): exr-sr-lowered-numeric, exr-sr-field-general.
- Exercise 14.6 (isotropic two-body decays boosted to the laboratory, opening angle, invariant mass): exr-sr-opening-angle, fig-sr-decay.

### part-i-11-lie-groups-algebras (textbook pass, task 4)

- Sec. 14.2.2, eq. (14.2) and the composition of rotations through the $j = 1/2$ representation (axis and angle from
  the eigenspinor): prp-lg-rotation-composition, exr-lg-comp-axis-angle (Sage).
- Secs. 14.2 and 14.4 (rotation matrices of every spin, rotating states and objects with them): def-lrep-rotation-matrix.
- Sec. 14.4.1 (a vector rotated through its spherical components with the $3 \times 3$ matrices): exm-lrep-dj-spherical-basis.
- Eq. (14.3) (exponentiation by diagonalisation, evenly spaced eigenvalues): exm-lrep-dj-diagonalise.
- Eq. (14.5) (spin vector $\langle s|J|s\rangle$): exr-lrep-dj-spin-vector.
- Exercise 14.1 (spherical harmonics from the middle column of $R^{(l)}$; restated with this book's z-x-z Euler angles,
  which shift the azimuth by $\pi/2$): exr-lrep-dj-harmonics (Sage).
- Exercise 14.3 (state $j = 3/2$, $m = 3/2$ rotated about the $x$-axis, probabilities and amplitudes over $[0, 4\pi]$; also
  $j = 2$): exr-lrep-dj-spin32 (Sage).

### part-ii-07-quantum-field-theory (textbook pass, task 4)

Sec. 24.2 eqs. 24.5-24.7, Sec. 24.4.4, Sec. 24.6 and Exercise 24.1 are used by part-ii-08-renormalisation, which these
blocks cross-reference instead of repeating.

- Sec. 24.2, eqs. (24.15)-(24.18) (free lattice propagator and the lattice dispersion relation
  $aE = 2\sinh^{-1}(\dots)$): prp-fpi-lattice-correlator, exm-fpi-lat-oscillator, exr-fpi-lat-images (Sage),
  exr-fpi-lat-residues.
- Sec. 24.2, "Spacetime symmetry" (hypercubic symmetry, $O(a^2)$ artefacts, rotation invariance recovered):
  exm-fpi-lat-artefacts.
- Sec. 24.2.2 (the mass gap $0.4949$ at $\kappa = 0.32$, $g = 0$, compared with the Monte Carlo $0.4965(5)$; eqs.
  (24.31)-(24.33), spectral decomposition and effective mass): exm-fpi-lat-gap, prp-fpi-spectral, exr-fpi-effective-mass
  (Sage).
- Eqs. (24.34)-(24.35) (one- and two-particle interpolating operators): exm-fpi-spectral-two-particle.
- Eqs. (24.9)-(24.12) (hopping-parameter form of the action; our $g = \kappa^2\lambda_0a^{4-d}/24$) and Exercise 24.4
  (the hopping parameter in the continuum limit, restated: $\kappa \to 1/d$ in the book's normalisation): exr-fpi-hopping,
  exr-fpi-mc-hopping-quartic, exm-fpi-mc-local.
- Exercise 24.5 (mass gap and correlation length near the critical coupling): exm-fpi-spectral-correlation-length.
- Sec. 24.2.1 (Metropolis-Hastings updates, eqs. (24.19)-(24.21); thermalisation and autocorrelation):
  def-fpi-metropolis, exm-fpi-mc-oscillator, exr-fpi-mc-oscillator (Sage); eq. (24.20): exm-fpi-mc-local; eq. (24.28)
  (over-relaxation by reflection): exr-fpi-mc-balance.
- Secs. 24.2 and 24.2.3 (Ising limit of $\phi^4$, parity breaking, $\kappa^* = 0.22166$ and $\kappa^*(g = 0.4) \approx 0.397$):
  exm-fpi-mc-ising-limit.

### part-ii-06-gauge-theories (physics foundations, task 3.1: lattice gauge theory)

Secs. 24.1 and 24.2 (scalar fields, Metropolis) are used by part-ii-07-quantum-field-theory and Sec. 24.6 (fermions) by
part-ii-08-renormalisation; the lattice section cross-references them. Our $\beta$ multiplies $1 - \frac1N\operatorname{Re}\operatorname{Tr}U_P$, so for
$\mathbb{Z}_2$ and $U(1)$ the book's $\beta'$ is $2\beta$; for $SU(2)$ the two agree.

- Sec. 24.3, eqs. (24.37)-(24.39) (Wegner's model, link variables, local $\mathbb{Z}_2$ transformations, Fig. 24.5, plaquette): def-lgt-link-variable,
  exm-lgt-link-z2, def-lgt-plaquette, exm-lgt-plaq-z2, def-lgt-z2, exm-lgt-wa-z2, exm-lgt-z2-orbit.
- Sec. 24.3, Elitzur's theorem, area and perimeter laws, $\beta^* = 1.5226$, eqs. (24.40)-(24.41): thm-lgt-elitzur, exm-lgt-el-ising,
  exm-lgt-z2-dual-3d, exm-lgt-z2-strong, exm-lgt-z2-weak, exm-lgt-al-z2-3d, exm-lgt-sc-z2, exm-lgt-wl-z2, exr-lgt-wa-conventions.
- Sec. 24.3.1 (heat bath, staples, eqs. (24.43)-(24.44), Fig. 24.6): def-lgt-staple, def-lgt-heat-bath, exm-lgt-st-z2, exm-lgt-hb-z2,
  exr-lgt-st-z2-table (Sage).
- Sec. 24.3.2 (average plaquette, Polyakov loop, hot and cold starts, Figs. 24.7-24.8): exm-lgt-hb-starts, exm-lgt-hb-z2-3d,
  exm-lgt-wl-polyakov, fig-lgt-z2-monte-carlo (our seeded heat bath on $6^3$).
- Sec. 24.4 and 24.4.1 (compact photons, eqs. (24.50)-(24.58), non-compact QED): exm-lgt-link-u1, def-lgt-compact-u1, exm-lgt-wa-u1,
  exm-lgt-wa-haar-abelian, exm-lgt-cl-u1, exm-lgt-cl-noncompact, exr-lgt-u1-noncompact-gauge.
- Sec. 24.4.2 (U(1) heat bath, eq. (24.59), coding considerations, memory estimate): exm-lgt-st-u1, exm-lgt-hb-u1, exm-lgt-st-hits,
  exm-lgt-link-count, exr-lgt-hb-u1-sampler (Sage).
- Sec. 24.4.3 (2D average plaquette eq. (24.60), Polyakov loop correlator, static potential eq. (24.62), $\sigma = 0.21076(7)$): prp-lgt-two-dim,
  exm-lgt-2d-u1, exm-lgt-2d-polyakov, def-lgt-wilson-loop, exm-lgt-wl-polyakov-correlator, exr-lgt-u1-string-2d (Sage),
  exr-lgt-wl-polyakov-loop, exr-lgt-wa-conventions.
- Sec. 24.4.4 (continuum limit, eq. (24.63), scaling windows): exm-lgt-cl-dimension, exm-lgt-2d-continuum-u1, exr-lgt-cl-2d-u1 (Sage).
- Sec. 24.5 (SU(2), eqs. (24.64)-(24.69), Haar measure): def-lgt-su2, exm-lgt-su2-quaternion, exm-lgt-su2-haar, exm-lgt-wa-su2,
  exr-lgt-su2-product (Sage), exr-lgt-su2-haar-norm.
- Sec. 24.5.1 (Creutz heat bath, eqs. (24.70)-(24.72), microcanonical updates, SU(2) product code): exm-lgt-su2-sum, exm-lgt-st-su2,
  exm-lgt-hb-creutz, exm-lgt-hb-overrelax, exr-lgt-hb-creutz-density.
- Sec. 24.5.2 (plaquette limits, string tension, eqs. (24.73)-(24.77), scaling form $e^{-6\pi^2\beta/11}$ with normalisation
  $e^{12\pi^2/11}$, the phases of gauge theory, Fradkin-Shenker, Elitzur and the Higgs field, glueball operators): exm-lgt-su2-limits,
  exm-lgt-su2-scaling, exm-lgt-wl-cornell, def-lgt-area-law, prp-lgt-strong-coupling, exm-lgt-sc-su2, exm-lgt-sc-u1-paradox,
  exm-lgt-sc-crossover, exm-lgt-al-su2, exm-lgt-al-linear, exm-lgt-al-higgs, exm-lgt-el-higgs-field, exm-lgt-cl-asymptotic-freedom,
  exr-lgt-al-scaling (Sage), exr-lgt-sc-su2-crossing (Sage), exr-lgt-int-su2 (Sage), fig-lgt-strong-coupling.
- Exercise 24.2 (U(1) string tension in four dimensions): exm-lgt-sc-u1, exm-lgt-u1-4d-phases, exr-lgt-sc-convention.
- Exercise 24.3 (Z2 string tension in two dimensions): exm-lgt-2d-z2, exr-lgt-2d-z2-sign.
- Exercise 24.6 (Z2 Polyakov loops, hot and cold starts): exm-lgt-hb-starts, exr-lgt-z2-polyakov.
- Exercise 24.7 (random gauge transformations): exr-lgt-link-random-gauge (Sage), exr-lgt-su2-random (Sage).
- Exercise 24.8 (temporal gauge, Polyakov loop): exm-lgt-link-temporal, exr-lgt-link-temporal-gauge.
- Exercise 24.9 (2D U(1) average plaquette): fig-lgt-two-dim, exr-lgt-int-2d-u1 (Sage).
- Exercise 24.10 (microcanonical SU(2) updates): exm-lgt-hb-overrelax, exr-lgt-hb-overrelax.
- Exercise 24.11 (Kennedy-Pendleton heat bath): exm-lgt-hb-kp, exr-lgt-hb-kp (Sage), exr-lgt-hb-acceptance (Sage).
- Exercise 24.14 (glueball operators, APE smearing, $\sigma = 0.18$ GeV$^2$): exm-lgt-su2-glueballs, exm-lgt-st-ape, exr-lgt-st-ape (Sage),
  exm-lgt-al-units, exr-lgt-wl-lattice-units (Sage), exr-lgt-al-force (Sage).
- Exercise 24.15 (SU(3) by SU(2) subgroups, plaquette limits): exm-lgt-su2-su3, exr-lgt-su2-equipartition.
- Exercise 24.16 (centre symmetry, Polyakov loop, centre vortices): exm-lgt-wa-symmetries, exm-lgt-su2-centre, exm-lgt-z2-centre,
  exm-lgt-el-polyakov, exr-lgt-su2-centre-symmetry.
- Exercise 24.17 (Z2 gauge-Higgs model, 4D Z2 transition at $\frac12\log(1 + \sqrt2)$): exm-lgt-z2-4d, exm-lgt-z2-gauge-higgs,
  exr-lgt-z2-gauge-higgs (Sage).
- Exercise 24.19 (Symanzik improvement): exm-lgt-cl-improvement.
- Errors found (Ch. 24):
  - eq. (24.47): $F_{\mu\nu} = \partial_\mu A_\nu - \partial_\nu A_\nu$ should read $\partial_\mu A_\nu - \partial_\nu A_\mu$;
  - eq. (24.53): the plaquette is $\exp(iea^2F_{\mu\nu})$, not $\exp(iea^2F_{\mu\nu}F_{\mu\nu})$;
  - p. 878, the SU(2) field strength $\frac{\sigma^a}{2}(\partial_\mu A^a_\mu - \partial_\nu A^a_\nu - \dots)$ has the derivative indices
    garbled: $\partial_\mu A^a_\nu - \partial_\nu A^a_\mu$;
  - eq. (24.69): the SU(2) Haar measure $\frac{1}{2\pi^2}\delta(a^2 - 1)d^4a$ integrates to $\frac12$; the normalised form is
    $\frac{1}{\pi^2}\delta(a^2 - 1)d^4a$ (eq. (24.58) for U(1) is correctly normalised) (exr-lgt-su2-haar-norm);
  - eq. (24.77): $\beta(g) = a\,\partial g/\partial a = -\frac{11N}{3(4\pi)^2}g^3 - \dots$ has the sign of $\mu\,\partial g/\partial\mu$;
    with the derivative in $a$ the coefficients are positive (the next relation $a\,dg_0/da = -\beta_0g_0^3$ with the negative
    $\beta_0$ is consistent);
  - Exercise 24.2: $\sigma = -\log(\beta/2) + \beta^2/8$ and "a phase transition near $\beta = 1$" hold for the weight $e^{\beta\cos\theta_P}$,
    not for the book's $e^{\frac\beta2\cos\theta_P}$ of eq. (24.55), where they read $-\log(\beta/4) + \beta^2/32$ and $\beta \approx 2$
    (exr-lgt-sc-convention);
  - Exercise 24.3: "$\sigma = \log\tanh(\beta)$" is negative; the 2D Z2 string tension is $-\log\tanh(\beta/2)$ in the book's
    normalisation (exr-lgt-2d-z2-sign);
  - Sec. 24.4.1: "a discretization in terms of $A_\mu(x)$ ... explicitly breaks U(1) gauge invariance" does not hold for the
    non-compact abelian theory with link angles, which is gauge invariant (exm-lgt-cl-noncompact, exr-lgt-u1-noncompact-gauge);
  - Sec. 24.3.2: "Eqs. 24.40 and 24.40" should be 24.40 and 24.41;
  - not an error, but eq. (24.41), $1 - 8e^{-4\beta'}$, is only the leading term: at $\beta' = 2$ the next term, $-88e^{-6\beta'}$ from
    vortex loops of six plaquettes, is a fifth of the first and is needed to match a Monte Carlo (exm-lgt-z2-weak,
    exr-lgt-z2-excitations).

