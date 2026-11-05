# A Modern Introduction to Classical Electrodynamics

Michele Maggiore, Oxford University Press, 2023. Local copy: `references/textbooks/A Modern Introduction to Classical Electrodynamics.pdf` (git-ignored). Textbook (design D32).

## Chapters it serves

Mapped from the book's table of contents (chapter titles and page numbers of the PDF). The sections, equations
and problem numbers actually used are added below when the chapter that uses them is written or revised.

| Chapter of this book | Sections of the textbook | Use |
|---|---|---|
| part-ii-04-classical-field-theory | Ch. 3 (Maxwell's equations, p. 56ff), Ch. 8 (covariant formulation, p. 200ff), Ch. 9 (waves in vacuum, p. 234ff), Ch. 10-11 (fields of moving charges, radiation) | content, examples and exercises |
| part-ii-06-gauge-theories | Ch. 8 (covariant formulation, gauge invariance, p. 200ff) | examples |
| part-ii-08b-special-relativity | Ch. 7 (special relativity, p. 155ff), Ch. 8 (covariant electrodynamics, p. 200ff), Sec. 9.6 | content |
| part-i-06-fourier-analysis | Ch. 6 (multipole expansion, spherical harmonics, p. 150ff) | examples and exercises |
| part-ii-01b-electromagnetism | Chs. 1-5 (statics, energy, forces), Secs. 6.2-6.4 (magnetic dipoles, multipoles in external fields) | content, examples and exercises (task 2.1) |
| part-ii-01b-electromagnetism | Ch. 3 (Maxwell's equations, conservation laws, potentials, gauge invariance, symmetries), Sec. 4.3 (induction), Sec. 8.6 (charged particle, as a pointer) | content, examples and exercises (task 2.2) |
| part-ii-01b-electromagnetism | Chs. 9-12 (waves, retarded potentials, Liénard-Wiechert fields, Larmor and dipole radiation, zones, radiation reaction) | content, examples and exercises (task 2.3) |
| part-ii-01b-electromagnetism | Chs. 13-16 (fields in matter, dispersion and Kramers-Kronig, waves in dielectrics, metals and waveguides, scattering) | content, examples and exercises (task 2.4) |

## Sections, equations and problems used

Chapter part-ii-04-classical-field-theory (task 12.8). The book uses the signature (-, +, +, +) and SI units; the blocks
below are converted to (+, -, -, -) and to the chapter's Heaviside-Lorentz units with c = 1 (SI where numbers are given).

- Sec. 8.3 (covariant Maxwell equations, charge conservation): exm-cfb-continuity.
- Sec. 8.4 (energy-momentum tensor of the field, radiation pressure): exm-cfb-radiation-pressure.
- Sec. 8.5 (transformation of E and B, invariants, crossed fields): exm-cfb-field-boost, exm-cfb-crossed-fields, exr-cfb-invariants (Sage).
- Sec. 8.6 (Lorentz force, interaction action of a point particle): exm-cfb-particle-action, exr-cfb-lorentz-force.
- Sec. 8.7.2 (Maxwell Lagrangian with sources): exm-cfb-source-coupling. Sec. 8.7.3 (Noether, symmetric tensor): exm-cfb-belinfante.
- Problem 8.1 (Lorentz invariance of the charge): exr-cfb-charge-invariance.
- Problem 8.2 (P^nu = int T^{0 nu} is a four-vector): exr-cfb-momentum-vector.
- Problem 8.3 (relativistic motion in a constant electric field): exr-cfb-constant-electric.
- Problem 8.4 (relativistic motion in a constant magnetic field): exr-cfb-constant-magnetic.
- Sec. 9.2 (Lorenz gauge, residual gauge freedom): exm-cfb-lorenz-gauge. Sec. 9.3 (Coulomb gauge): exr-cfb-coulomb-gauge.
- Sec. 9.5 (polarisation, Stokes parameters, circular basis): exm-cfb-polarisation, exr-cfb-stokes (Sage), exr-cfb-circular-basis.
- Sec. 9.6 (Doppler effect and aberration): exm-cfb-doppler, exr-cfb-aberration (Sage).
- Sec. 10.3 (field of a charge in uniform motion): exm-cfb-moving-charge, exr-cfb-boosted-coulomb.
- Sec. 10.5 (Larmor formula): exm-cfb-larmor. Sec. 10.6 (relativistic generalisation): exm-cfb-lienard, exr-cfb-synchrotron-loss.
- Problem 10.1 (oscillating dipole): exr-cfb-dipole-power (Sage).
- Problem 10.2 (charge on a circular orbit): exr-cfb-circular-orbit (Sage).
- Problem 10.3 (radiative collapse of the classical hydrogen atom): exr-cfb-atom-collapse (Sage).
- Problem 11.1 (transverse and longitudinal parts of a vector field): exm-cfb-helmholtz.
- Sec. 11.3 (near, induction and far zones): exm-cfb-zones.
- Not used (task 8 check): Ch. 3 (Maxwell's equations in vector form, p. 56ff), since the chapter starts from the
  covariant formulation of Ch. 8 and recovers the vector equations from it; the remaining sections of Chs. 9-11
  (waves in matter, radiation of extended sources) go beyond the chapter's statements.

Chapter part-ii-08b-special-relativity (task 12.9). Maggiore's signature (-, +, +, +) is converted by $\eta \to -\eta$; his boost
(7.19) is the active boost with $v_0 \to -v_0$ relative to ours.

- Secs. 7.1-7.2.1 (postulates; invariance of the interval by the Landau-Lifshitz argument, eqs. (7.3)-(7.12)): def-sr-inertial-frame,
  exm-sr-frame-uniform, prp-sr-interval (our proof replaces the three-frame argument by the quadratic-form lemma and $a^2 = 1$).
- Sec. 7.2.1, eqs. (7.19)-(7.31) (hyperbolic rotations, rapidity, composition of velocities): def-sr-rapidity, prp-sr-velocity-addition,
  exm-sr-sector-area.
- Secs. 7.2.2-7.2.4 (causality, proper time, contraction): def-sr-minkowski-spacetime, prp-sr-dilation-contraction, def-sr-proper-time.
- Sec. 7.3 (the Lorentz group), cited in the introduction of the part on the Lorentz group (before def-sr-o11). Sec. 7.3.1
  (det = $\pm1$, proper transformations): prp-sr-det-lambda. Sec. 7.3.4, eq. (7.86) ($\omega_{\mu\nu} = -\omega_{\nu\mu}$): exm-sr-lie-count.
- Sec. 7.3.5 (decomposition of a Lorentz tensor under rotations): exm-sr-tensor-rotations.
- Sec. 7.3.6, eqs. (7.92)-(7.114) (transformation of fields, gradients, divergences; note 17): def-sr-field-transformation,
  exm-sr-field-radial, exr-sr-gradient, exr-sr-divergence-scalar.
- Sec. 7.4 (relativistic mechanics), cited for the dynamics section: def-sr-four-force. Sec. 7.4.2 (action of a free
  particle): prp-sr-particle-action. Secs. 8.5-8.6.1 (field transformation, covariant Lorentz
  force): prp-sr-field-transformation, prp-sr-lorentz-force. Sec. 9.6 (Doppler, aberration): def-sr-wave-vector.
  Sec. 10.3 (field of a charge in uniform motion): exm-sr-moving-charge.

Chapter part-ii-06-gauge-theories (textbook pass). Signature converted from (-, +, +, +).

- Sec. 8.6.2, eqs. (8.68)-(8.78) (interaction action of a point charge, canonical momentum P = p + qA): exm-gth-canonical-momentum.
- Sec. 8.6.2, eq. (8.75) (gauge invariance of the coupling to a current requires its conservation): exr-gth-source-gauge.

Chapter part-i-06-fourier-analysis (textbook pass, task 5). Written for $\Phi = \int\rho/|\mathbf x - \mathbf x'|$; the
potential is $\Phi/4\pi$ in Heaviside-Lorentz and $\Phi/4\pi\varepsilon_0$ in SI units.

- Sec. 6.1, eqs. (6.5)-(6.20), (6.26)-(6.29) (monopole, dipole, quadrupole, STF moments; here in spherical harmonics, the
  relation the book leaves to note 4): prp-fam-multipole, exm-fam-point-charges.
- Sec. 6.1, eqs. (6.21)-(6.25) (dependence on the origin): exm-fam-origin; eq. (6.24) (quadrupole under a shift): exr-fam-quadrupole-shift.
- Sec. 6.1, eqs. (6.11)-(6.13) (dipole field): exm-fam-dipole-field.
- Sec. 6.1, Fig. 6.3 (charge distributions with a pure dipole and a pure $Q_{xz}$): exr-fam-sphere-moments (Sage).
- Secs. 6.4.1-6.4.2, eqs. (6.59)-(6.70), (6.88)-(6.89) (multipoles in an external field, torque, dipole-dipole energy):
  exr-fam-external-field.
- Not used: Sec. 6.2-6.3 (magnetic multipoles, point dipoles) and Problem 6.1 (Larmor precession), which belong to the
  classical field theory chapter.

Chapter part-ii-01b-electromagnetism, section "Electrostatics and magnetostatics" (change kk-csdr-physics-foundations,
task 2.1). The book's SI formulas are converted to Heaviside-Lorentz units with $c = 1$ (SI where numbers are given).
Maggiore starts from Maxwell's equations (top-down); the section is written bottom-up from Coulomb's and the
Biot-Savart laws, since Maxwell's equations come in the next section, and the static field equations are proved from
the integral laws with the Green's function of the Laplacian.

- Secs. 4.1.1-4.1.2, eqs. (4.4)-(4.20) (Coulomb's law, field of a static density): def-ems-electric-field.
- Secs. 3.1.2 and 4.1 (integral and local Gauss law, curl-free field): thm-ems-gauss-law.
- Sec. 4.1.2, eqs. (4.26)-(4.27) (force between two distributions, third law): exm-ems-third-law.
- Sec. 4.1.3, eqs. (4.16), (4.30)-(4.33) (electrostatic potential, work, conservative force): def-ems-electrostatic-potential.
- Sec. 4.1.4 (Earnshaw's theorem): exm-ems-earnshaw.
- Sec. 4.1.5 and Sec. 4.1.6, eq. (4.55) (uniqueness, Dirichlet, Neumann and fixed-charge conditions): prp-ems-uniqueness;
  eqs. (4.45)-(4.47) (a field from its divergence and curl): exm-ems-uniqueness-vector.
- Sec. 4.1.6 (conductors, field at the surface, Faraday cage): def-ems-conductor, exm-ems-gauss-jump, exm-ems-uniqueness-cavity.
- Sec. 4.1.7 (forces from surface integrals of the stress tensor, two equal charges): exr-ems-mech-stress (Sage).
- Sec. 4.2.1 (straight wire, symmetry argument; note 22, vector potential of the wire): exm-ems-wire-field,
  exm-ems-ampere-symmetry, exm-ems-A-wire.
- Secs. 4.2.2-4.2.4 (Biot-Savart law, force on currents, parallel wires): def-ems-magnetic-field, thm-ems-ampere-law,
  def-ems-vector-potential, exm-ems-parallel-wires.
- Problem 4.1 (charged plane): exm-ems-plane. Problem 4.2 (spherical distribution): exm-ems-gauss-ball.
- Problems 4.3, 4.4 (parallel-plate and spherical capacitors): def-ems-capacitance, exm-ems-capacitance-plates,
  exm-ems-capacitance-spherical.
- Problem 4.5 (Madelung energy of NaCl): exm-ems-energy-madelung, exr-ems-energy-madelung (Sage).
- Problem 4.6 (solenoid): exm-ems-solenoid-axis.
- Secs. 5.1-5.4 (work and energy in electrostatics and magnetostatics, field energy, conductors, inductances):
  prp-ems-field-energy; Sec. 5.2.2 (self-energy, classical electron radius): exm-ems-energy-self.
- Problem 5.1 (energy of a capacitor): exm-ems-energy-capacitor. Problem 5.2 (Green's reciprocity): exm-ems-capacitance-reciprocity.
- Secs. 5.5, 5.5.1-5.5.2 (mechanical potentials at fixed charge, fixed potential, fixed current): prp-ems-mechanical-potentials,
  exr-ems-mech-legendre, exm-ems-mech-wires; Sec. 6.4 (multipoles in external fields): prp-ems-mechanical-potentials (d).
- Secs. 6.2-6.3 (magnetic multipoles, point dipoles): def-ems-magnetic-dipole. Sec. 6.1 (electric multipoles) is used by
  cross-reference to prp-fam-multipole of the Fourier chapter.
- Not used: Ch. 1 (mathematical tools, in Part I), Ch. 2 (systems of units, summarised in the section's units
  paragraph), Ch. 3 and Sec. 4.3 (Maxwell's equations, induction: task 2.2), Problem 4.7 (Ohmic dissipation), Problem 4.8
  and Problem 5.3 (inductance from induction: task 2.2).
- Errors found: Sec. 4.1.5, last paragraph (p. 65), "In electrostatics, we have $\nabla\cdot\mathbf E = -\rho/\varepsilon_0$":
  the sign is wrong, Gauss's law (3.1) is $+\rho/\varepsilon_0$. Sec. 4.1.4 (p. 63) argues that at a minimum of $\phi$
  "$\mathbf v\cdot\nabla\phi$ is strictly positive in $\mathbf x$" for every $\mathbf v$, which is impossible (the gradient
  vanishes at an interior minimum); the argument needs the outward normal derivative to be positive on a small sphere
  around a strict minimum, which is how exm-ems-earnshaw uses the maximum principle instead.

Chapter part-ii-01b-electromagnetism, section "Maxwell's equations, potentials and gauge invariance" (change
kk-csdr-physics-foundations, task 2.2). SI converted to Heaviside-Lorentz units with $c = 1$; Maggiore's gauge function
$\theta$ of eq. (3.86) is $-\lambda$ in our convention (that of the Dirac and gauge-theory chapters).

- Secs. 3.1.2 and 4.3 (integrated Faraday law, Lenz's law, moving loops): def-emm-induction, exm-emm-lenz (Sec. 4.3.1),
  exm-emm-flux-rule (Sec. 4.3.2).
- Secs. 3.1-3.2.1 (Maxwell's equations, displacement current, continuity): def-emm-maxwell-equations,
  exm-emm-charge-conservation; eqs. (3.25)-(3.30) (point charges): exm-emm-point-currents; Sec. 3.1.2 (integral forms):
  exm-emm-integral-forms; Ch. 2 and Sec. 3.1.1 (units): exm-emm-si-form.
- Sec. 3.2.2, eqs. (3.31)-(3.47) (Poynting's theorem): prp-emm-poynting; note 11 (non-uniqueness of u and S):
  exm-emm-poynting-ambiguity; eqs. (3.49)-(3.55) (relativistic particles): exm-emm-relativistic-work.
- Sec. 3.2.2, eqs. (3.56)-(3.79) (momentum density, Maxwell stress tensor in the sign convention of note 14, angular
  momentum): prp-emm-field-momentum, exm-emm-momentum-energy; eqs. (3.70)-(3.72): exm-emm-force-density; Secs. 4.1.7 and
  4.2.5 (forces from surface integrals): exm-emm-stress-static.
- Sec. 3.3, eqs. (3.80)-(3.85) (potentials): def-emm-potentials, exm-emm-pot-counting, exm-emm-pot-existence,
  exm-emm-constraints; eq. (3.86) (gauge transformation): def-emm-gauge-transformation; eqs. (3.87)-(3.94) (Lorenz and
  Coulomb gauges): prp-emm-gauges, exm-emm-coulomb-instantaneous; the remark on the role of the potentials:
  exm-emm-gi-why-potentials.
- Sec. 3.4, eqs. (3.95)-(3.99) (translations, rotations, parity, time reversal, duality): prp-emm-symmetries,
  exm-emm-sym-potentials, exm-emm-sym-noether; Sec. 4.2.1 (parity argument for the wire): exm-emm-sym-wire-parity.
- Sec. 8.6, 8.6.2 (charged particle, interaction action, canonical momentum), cited as a pointer only, since the covariant
  form stays in the later chapters: def-emm-minimal-coupling, prp-emm-gauge-invariance-motion.
- Not used in task 2.2: Sec. 3.2.2 on angular momentum beyond the statement in prp-emm-field-momentum; Problems of
  Ch. 3 (none in the book); Sec. 8.7 (Lagrangian of the field: classical field theory chapter).
- Errors found: none in Ch. 3.

Chapter part-ii-01b-electromagnetism, section "Waves, moving charges and radiation" (task 2.3). Vector form, Heaviside-Lorentz
units with $c = 1$; the covariant blocks of the classical field theory and special relativity chapters (cfb-, sr-) now refer back
to these blocks.

- Secs. 9.1-9.4 (wave equations, plane waves, Fourier decomposition): def-emw-plane-wave, exm-emw-pw-fourier.
- Sec. 9.5 (polarisation, Stokes parameters): def-emw-polarisation, exm-emw-pol-circular, exm-emw-pol-measure.
- Secs. 10.1 and 9.2 (retarded and advanced Green's functions, Lorenz gauge): prp-emw-retarded-potentials, exm-emw-ret-advanced;
  Sec. 11.1 (far zone): exm-emw-ret-far; Sec. 12.1 (expansion in the retardation): exm-emw-ret-expansion.
- Secs. 10.2-10.4 (Liénard-Wiechert potentials and fields, uniform motion, radiation field): prp-emw-lienard-wiechert,
  exm-emw-ret-point, exm-emw-lw-uniform, exm-emw-lw-two-parts, exm-emw-lw-slow; Problem 10.2 (charge on a circle): exm-emw-lw-circle.
- Secs. 10.5-10.6 (Larmor, Liénard, accelerations along and across the velocity, beaming): prp-emw-larmor, exm-emw-larmor-lienard,
  exm-emw-larmor-beaming, exm-emw-lw-beaming.
- Secs. 11.2-11.2.2 (dipole, magnetic dipole and quadrupole radiation): prp-emw-dipole-radiation, exm-emw-dip-quadrupole; Problem
  10.1 (oscillating dipole): exm-emw-dip-oscillating.
- Sec. 11.3 (near, induction and far zones): def-emw-radiation-zones, exm-emw-zones-induction.
- Secs. 12.3.3-12.3.5 (radiation reaction, Schott term, Abraham-Lorentz-Dirac): prp-emw-radiation-reaction, exm-emw-rr-energy,
  exm-emw-rr-ald.
- Not used in task 2.3: Sec. 9.6 (Doppler effect and aberration, kept in the covariant chapters), Sec. 11.1.2-11.1.3 (Coulomb-gauge far
  fields, spectral distribution), Secs. 12.2 (1PN dynamics) and 12.3.1-12.3.2 (extended electron models, mass renormalisation).
- Errors found: none.

Chapter part-ii-01b-electromagnetism, section "Fields in matter" (task 2.4), and the chapter's closing Exercises section. Vector
form, Heaviside-Lorentz units with $c = 1$ (SI where numbers are given); relative permittivities and permeabilities of SI tables
are used unchanged.

- Secs. 13.1-13.2 (macroscopic averaging, free and bound charges): def-emx-polarisation, exm-emx-pol-averaging.
- Sec. 13.3 (macroscopic current density, magnetisation and polarisation currents): def-emx-magnetisation.
- Secs. 13.4-13.5 (Maxwell's equations in media, boundary conditions on E, B, D, H): prp-emx-macroscopic-maxwell.
- Sec. 13.6 (constitutive relations: dielectrics, metals, dia- and paramagnets): def-emx-linear-medium; Sec. 13.6.3:
  exm-emx-lin-permeabilities.
- Sec. 13.7 (energy conservation in media): prp-emx-energy-media.
- Secs. 14.1-14.2 (general properties of the response, causality, Kramers-Kronig): prp-emx-kramers-kronig, exm-emx-kk-conductor,
  exm-emx-kk-sum-rule.
- Secs. 14.3-14.5 (Drude-Lorentz model, Drude conductivity, dielectric function of metals): def-emx-drude-lorentz,
  exm-emx-kk-line, exm-emx-dl-conductivity, exm-emx-dl-transparency.
- Secs. 15.1-15.3 (waves in dielectrics, phase and group velocity, transverse and longitudinal waves in metals):
  prp-emx-waves-media, exm-emx-wv-plasma, exm-emx-wv-anomalous, exm-emx-wv-metal-mirror, exm-emx-wv-longitudinal.
- Sec. 15.4 (waveguides: boundary conditions, TE, TM and TEM modes): prp-emx-waveguide.
- Secs. 16.1-16.3 (cross-section, Thomson and bound-electron scattering): prp-emx-scattering, exm-emx-sc-thomson.
- Not used in task 2.4: Sec. 13.8 (solved problems, covered by the problem books), the microscopic derivation of the averaging
  beyond exm-emx-pol-averaging, and the detailed band-structure remarks of Sec. 14.5.
- Errors found: none in Chs. 13-16.
