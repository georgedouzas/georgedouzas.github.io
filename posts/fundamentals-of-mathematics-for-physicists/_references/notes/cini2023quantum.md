# Quantum Mechanics

Michele Cini, Stefano Forte, Guido Montagna, Oreste Nicrosini, Luca Peliti and Alberto Rotondi, Springer, 2023. Local copy: `references/textbooks/Cini et al. - Quantum Mechanics.pdf` (git-ignored). Textbook (design D32).

## Chapters it serves

Mapped from the book's table of contents (chapter titles and page numbers of the PDF). The sections, equations
and problem numbers actually used are added below when the chapter that uses them is written or revised.

| Chapter of this book | Sections of the textbook | Use |
|---|---|---|
| part-ii-03-quantum-mechanics | Ch. 3-6 (axioms, representations, uncertainty, oscillator, p. 64-125), Ch. 9-14 (one-dimensional potentials, central potentials, hydrogen, scattering, p. 190-307) | content, examples and exercises |
| part-i-07-hilbert-spaces-operators | Ch. 2 (linear vector spaces, p. 28ff) | examples |
| part-i-05-differential-equations | Ch. 7 (theory of second-order differential equations, p. 126ff) | examples and exercises |

## Sections, equations and problems used

Chapter part-i-07-hilbert-spaces-operators (Functional analysis):

- Sec. 2.2.3 (a "complete" set of vectors as a spanning set) and Sec. 2.8 (Cauchy sequences, completeness, completeness of
  finite-dimensional spaces, pp. 39-41): @def-banach-space, @exm-fan-finite-dim-complete, @exr-fan-two-completes.

Chapter part-ii-05-dirac-equation (The Dirac equation):

- Sec. 11.2, eqs. (11.5)-(11.14) (separation of the hydrogen problem, the asymptotic factors, the terminating series and the
  quantisation condition): the method of the proof of thm-dq-dirac-levels; eq. (11.6) (radial equation for r R): exm-dq-second-order;
  eq. (11.17) (normalised radial functions, value at the origin): exr-dq-darwin-expectation.
- Sec. 11.3, Problem 2 (<r> for the ground state): exr-dq-cini-mean-r; Problem 3 (<1/r>, potential and kinetic energy):
  exr-dq-fine-structure-sum, exr-dq-virial.
- Sec. 13.2 (square barrier, continuity of psi and psi'): exr-dq-klein-matching.
- Sec. 13.4, Problem 1 (step of infinite extent, reflection for E < V0 and E > V0): exm-dq-klein-nonrel, exr-dq-cini-step and the
  Schrodinger curve of fig-dq-klein.

Chapter part-ii-03-quantum-mechanics (task 12.8), cited as `[@cini2023quantum, Sec. N.M, Problem n]`:

- Sec. 3.6, Problem 2 (oscillator state: normalisation, energy probabilities, mean energy, repeated measurement, time
  evolution): exm-qmb-oscillator-measurement, exr-qmb-oscillator-state (Sage).
- Sec. 3.6, Problem 4 (complex potential, source and sink of probability): exr-qmb-absorbing-potential.
- Sec. 4.7, Problem 1 (anticommuting involutions are traceless): exr-qmb-anticommuting.
- Sec. 4.7, Problem 2 (constant shift of the potential): exm-qmb-constant-shift.
- Sec. 4.7, Problem 3 (wave function vanishes where V is infinite): exm-qmb-infinite-wall.
- Sec. 4.7, Problem 6 (eigenvectors of the 3 x 3 matrix, here S_x of spin one): exm-qmb-spin-one-matrix.
- Sec. 4.7, Problem 7 (general two-state Hamiltonian): exm-qmb-two-state-hamiltonian.
- Sec. 4.7, Problem 9 (Heisenberg equations of the free particle): exm-qmb-heisenberg-free.
- Sec. 4.7, Problem 10 (Heisenberg equations for V = kx^2/2 + beta x): exr-qmb-linear-oscillator.
- Sec. 5.3, Problems 1-2 (i[A, B] Hermitian; conserved expectation values): exr-qmb-commutator-hermitian.
- Sec. 5.3, Problem 3 (L_x, L_y in |l, 0>): exm-qmb-uncertainty-lm, exr-qmb-lx-ly (Sage).
- Sec. 6.6, Problems 3 and 5 (matrix elements of powers of x): exm-qmb-x-cubed.
- Sec. 8.7, Problems 1-2 (a one-dimensional attractive potential always binds): exm-qmb-weak-binding, exr-qmb-bound-state-1d.
- Sec. 8.7, Problem 3 (number of bound states of the finite well): exm-qmb-finite-well-count, exr-qmb-finite-well (Sage).
- Sec. 8.7, Problem 4 (virial theorem): exr-qmb-virial.
- Sec. 9.8, Problem 1 (box on [0, 2a]): exm-qmb-shifted-box. Problem 2 (finite well without the symmetry): exr-qmb-finite-well.
- Sec. 9.8, Problem 4 (equal superposition of three oscillator states): exr-qmb-three-state-packet (Sage).
- Sec. 10.4, Problems 1-2 (operators commuting with L_x, L_y; only l = 0 has all components sharp): exr-qmb-sharp-components.
- Sec. 10.4, Problem 3 (commutators of L_+): exm-qmb-ladder-relations, exr-qmb-ladder-commutators.
- Sec. 11.3, Problems 1-2 (radial densities, <r> and <r^2> for 1s, 2s, 2p): exm-qmb-hydrogen-radial, exr-qmb-hydrogen-moments (Sage).
- Sec. 12.5, Problem 2 (<r>, <r^2> for the isotropic oscillator): exm-qmb-isotropic-radius, exr-qmb-isotropic-excited.
- Sec. 13.2 (rectangular barrier, transmission): exm-qmb-tunnelling-microscope, exr-qmb-barrier (Sage).
- Sec. 14.10, Problem 3 (s-wave scattering by a spherical barrier; hard-sphere limit): exm-qmb-hard-sphere, exr-qmb-swave (Sage).

Chapter part-i-05-differential-equations (textbook pass, task 6):

- Sec. 7.1.2 (linear dependence, the Wronskian, Property 3: at most two independent solutions): prp-wronskian,
  exm-wr-three-solutions, exr-wr-smooth (the remark on isolated zeros of the Wronskian), exr-wr-variation,
  exr-wr-liouville (Sage), exr-frob-legendre-second (Sage).
- Sec. 7.1.1 (ordinary, regular and irregular singular points; the point at infinity): def-regular-singular-point,
  exm-frob-legendre-points, exm-frob-bessel-points; Problem 2 of Sec. 7.3 (y'' + w^2 y = 0 at infinity):
  exm-frob-oscillator-infinity; Problem 8(d) (hypergeometric equation): exr-frob-hypergeometric; Problem 6(a) (Euler-type
  equation y'' - b(b + 1)y/x^2 = 0): exr-frob-euler.
- Sec. 7.1.3, eqs. (7.10)-(7.18) and the five observations after them (Frobenius method, indicial equation, the cases
  of equal and integer-spaced indices): thm-frobenius-method, exm-frob-oscillator (with Problems 3-5),
  exm-frob-bessel, exr-frob-bessel-log (Sage).
- Secs. 7.1.3-7.1.4 (termination of the series as the quantisation condition): exm-frob-legendre-termination; Sec. 7.2
  (Hermite equation, termination for integer nu): exr-frob-hermite (Sage); Sec. 7.1.5 (radial equation): exr-frob-radial.
- Errors noted: Sec. 7.1.2 calls analytic functions "differentiable any number of times" and states that a vanishing
  Wronskian implies dependence; this holds for analytic functions and for solutions of one equation with continuous
  coefficients, not for smooth functions (x^3 and |x|^3, exr-wr-smooth).
- Not used: Sec. 7.1.4 (Sturm-Liouville theory) and the rest of Sec. 7.2 (standard equations), which the chapter
  already treats in its Sturm-Liouville, Legendre and Bessel parts.
