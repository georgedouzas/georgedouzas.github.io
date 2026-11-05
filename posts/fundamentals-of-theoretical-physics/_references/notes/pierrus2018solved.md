# Solved Problems in Classical Electromagnetism

J. Pierrus, Oxford University Press, 2018. Local copy: `references/textbooks/Pierrus - 2018 - Solved Problems in Classical Electromagnetism Ana.pdf` (git-ignored). Textbook (design D32), from the author's Zotero library, which was copied in full
for change kk-csdr-physics-foundations and later changes (design E5).

## Chapters it serves

Mapped from the book's table of contents. The sections, equations and problem numbers actually used are added below
when the chapter that uses them is written or revised.

| Chapter of this book | Sections of the textbook | Use |
|---|---|---|
| part-ii-01b-electromagnetism | Ch. 2-3 (static electric fields, conductors), Ch. 4 (static magnetic fields) used in task 2.1; Ch. 5 (quasi-static fields) for task 2.2 | exercises (task 2.1) |
| part-ii-01b-electromagnetism | Ch. 5 (quasi-static fields), Ch. 7 (Maxwell's equations, energy), Ch. 8 (the potentials, charged particle) used in task 2.2; Ch. 7 (waves) and Ch. 11 (radiation) for task 2.3 | exercises (tasks 2.2, 2.3) |
| part-ii-01b-electromagnetism | Ch. 9-10 (fields in matter, Maxwell's equations in matter) and Questions 7.12-7.13, 7.15 (wave guides) used in task 2.4 | exercises and Sage exercises (task 2.4) |
| part-ii-08b-special-relativity | Ch. 12 (electromagnetism and special relativity) | exercises |

## Sections, equations and problems used

Chapter part-ii-01b-electromagnetism, section "Electrostatics and magnetostatics" (change kk-csdr-physics-foundations,
task 2.1). The book uses SI units and Mathematica; the blocks use Heaviside-Lorentz units with $c = 1$ (SI for numbers)
and Sage.

- Question 2.4 (charged shell): exm-ems-gauss-shell. Question 2.5 (disc on its axis): exm-ems-disc-axis,
  exr-ems-potential-disc (Sage). Question 2.7 (sphere with a cavity): exm-ems-gauss-cavity.
- Question 2.8 (line charge): exm-ems-line-charge. Question 2.9 (finite line): exr-ems-segment (Sage).
- Question 2.10 (ring): exm-ems-ring-axis. Question 2.11 (two charges): exm-ems-two-charges.
- Question 2.14 (lines of force): exm-ems-field-lines. Question 2.15 (dipole lines and equipotentials): exr-ems-dipole-lines (Sage).
- Question 2.16 (two line charges): exm-ems-potential-two-lines.
- Question 3.4 (coaxial cylinders): exm-ems-uniqueness-coax. Questions 3.11-3.12 (coefficients of potential and
  capacitance; concentric shells): def-ems-capacitance, exm-ems-capacitance-concentric, exr-ems-capacitance-three-shells (Sage).
- Question 3.13 (two distant spheres): exm-ems-capacitance-two-spheres, exr-ems-capacitance-two-spheres.
- Questions 3.16-3.17 (box with a live lid, truncated series): exm-ems-uniqueness-box, exr-ems-uniqueness-cube-centre (Sage).
- Question 3.23 (charge near a grounded sphere): exm-ems-image-sphere.
- Question 4.4 (force between circuits): exr-ems-circuits-third-law. Questions 4.5-4.6 (magnetostatic equations from the
  Biot-Savart law, Coulomb gauge): thm-ems-ampere-law, def-ems-vector-potential, exm-ems-A-coulomb-gauge.
- Question 4.7 (thick wire): exm-ems-ampere-wire. Question 4.8 (loop axis, straight segment): exm-ems-loop-axis,
  exm-ems-segment-field. Question 4.9 (solenoid): exm-ems-solenoid-axis.
- Question 4.11 (spinning charged sphere): exr-ems-dipole-ball (Sage). Question 4.12 (wire with a hole): exm-ems-ampere-hole.
- Questions 4.16-4.17 (coaxial coils with equal and opposite currents): exr-ems-helmholtz, exr-ems-anti-helmholtz (Sage).
- Question 4.25 (moment of a plane loop): exm-ems-dipole-planar, exr-ems-dipole-planar-proof.
- Question 4.29 (magnet on the axis of a coil): exr-ems-mech-coil-magnet (Sage).
- Not used in task 2.1: Questions 2.1-2.3 (delta-function densities, in Part I), 2.12-2.13, 2.17-2.30 (energy of point
  charges and multipoles, partly covered by Maggiore and the Fourier chapter), 3.1-3.3, 3.5-3.10, 3.14-3.15, 3.18-3.22,
  3.24-3.34 (numerical relaxation, conformal maps and further image problems), 4.1-4.3, 4.10, 4.13-4.15, 4.18-4.24,
  4.26-4.28, 4.30 (solid angle, charged particles in dipole fields, multipole expansions of currents).
- Errors found: none in the questions used.

Chapter part-ii-01b-electromagnetism, section "Maxwell's equations, potentials and gauge invariance" (task 2.2).

- Question 5.1 (quasi-static limit): exm-emm-quasistatic. Question 5.6 (falling rail): exr-emm-falling-bar (Sage).
- Question 5.7 (rotating coil with R and L): exm-emm-generator, exr-emm-generator-current (Sage).
- Question 5.12 (field induced by a growing field in a cylinder): exm-emm-induced-cylinder, exr-emm-induced-check (Sage).
- Question 5.13 (particle in a time-dependent field; betatron): exm-emm-betatron.
- Question 7.1 (continuity equation): exm-emm-charge-conservation.
- Questions 8.1-8.2 (potentials, second-order equations): def-emm-potentials. Question 8.3 (point charge with a vector
  potential only): exm-emm-gauge-point-charge, exr-emm-pot-point-charge (Sage). Question 8.5 (gauge invariance of the
  flux): exr-emm-pot-flux-invariance. Questions 8.6-8.7 (residual freedom in the Coulomb and Lorenz gauges): prp-emm-gauges.
- Questions 8.16-8.18 (Lagrangian and Hamiltonian of a charge, weak-field expansion): def-emm-minimal-coupling,
  exm-emm-mc-lorentz, exm-emm-mc-weak-field, exr-emm-mc-weak-field, prp-emm-gauge-invariance-motion.
- Not used in task 2.2: Questions 5.2-5.5, 5.8-5.11, 5.14-5.21 (inductance computations, moving magnets), 7.2-7.16
  (waves, wave guides: task 2.3), 8.8-8.15 (retarded potentials, Liénard-Wiechert: task 2.3), 8.19-8.20.
- Errors found: none in the questions used.

Chapter part-ii-01b-electromagnetism, section "Waves, moving charges and radiation" (task 2.3).

- Questions 7.4-7.5 (plane waves in vacuum, transversality): def-emw-plane-wave, exm-emw-pw-transverse.
- Questions 8.8-8.9 (retarded potentials, time-harmonic potentials): prp-emw-retarded-potentials.
- Questions 8.12-8.14 (point charge in arbitrary motion, Liénard-Wiechert fields): prp-emw-lienard-wiechert. Question 8.15 (uniform
  motion): exm-emw-lw-uniform, exm-emw-lw-uniform-potential.
- Questions 11.12-11.13 (loop antenna and short dipole antenna): exm-emw-dip-magnetic, exm-emw-dip-antenna.
- Not used in task 2.3: Questions 7.6-7.16 (energy theorems and wave guides, partly task 2.4), 8.10-8.11, 11.1-11.11, 11.14-11.24.
- Errors found: none in the questions used.

Chapter part-ii-01b-electromagnetism, section "Fields in matter" (task 2.4).

- Questions 9.1-9.5 (polarisation, bound charges, polarised rod and slab, Langevin): def-emx-polarisation, exm-emx-mm-capacitor,
  exr-emx-pol-potential, exr-emx-pol-rod, exm-emx-pol-slab, exm-emx-pol-langevin. Questions 9.6-9.9 (dielectric sphere, shell,
  capacitor energy, refraction of lines): exm-emx-mm-sphere, exm-emx-mm-shell, exm-emx-en-capacitor, exm-emx-mm-refraction.
- Questions 9.11-9.12 (energy of a dielectric, molecular picture): prp-emx-energy-media, exm-emx-en-polarisation-work,
  exr-emx-en-molecular.
- Questions 9.13, 9.16, 9.19-9.20 (magnetisation, bound currents, magnetised sphere, H inside a magnet): def-emx-magnetisation,
  exr-emx-mag-potential, exr-emx-mag-gap, exm-emx-mag-sphere, exr-emx-mag-sphere-check (Sage), exm-emx-mm-h-inside. Questions
  9.17-9.18 (iron ring with a gap): exm-emx-mm-iron-ring, exr-emx-mm-electromagnet (Sage). Question 9.22 (inductor with a core):
  exm-emx-en-inductor.
- Questions 10.1-10.2 (polarisation current, Maxwell's equations in matter): def-emx-polarisation, exm-emx-pol-current. Question
  10.6 (boundary conditions): prp-emx-macroscopic-maxwell. Question 10.8 (Drude model of copper): def-emx-drude-lorentz,
  exm-emx-dl-copper. Question 10.9 (relaxation of free charge): exm-emx-mm-free-charge. Questions 10.10-10.11 (waves in conductors,
  skin depth): prp-emx-waves-media, exm-emx-wv-skin, exr-emx-wv-skin (Sage).
- Questions 7.12-7.13 and 7.15 (rectangular wave guide, TE10 mode, cut-offs): prp-emx-waveguide, exm-emx-wg-te10, exr-emx-wg-band (Sage).
- Not used in task 2.4: Questions 9.10, 9.14-9.15, 9.21, 9.23, 10.3-10.5, 10.7, 10.12-10.20, and Appendix J (the Sage
  solutions of the chapter are written independently).
- Errors found: none in the questions used.
