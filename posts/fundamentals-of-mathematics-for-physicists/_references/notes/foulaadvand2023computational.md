# A Computational Approach to Physics

M. Ebrahim Foulaadvand, Cambridge Scholars Publishing, 2023. Local copy: `references/textbooks/A Computational Approach to Physics.pdf` (git-ignored). Textbook (design D32).

## Chapters it serves

Mapped from the book's table of contents (chapter titles and page numbers of the PDF). The sections, equations
and problem numbers actually used are added below when the chapter that uses them is written or revised.

| Chapter of this book | Sections of the textbook | Use |
|---|---|---|
| part-i-05-differential-equations | Ch. 3 (oscillatory motion, p. 60ff), Ch. 5-7 (parabolic, hyperbolic and elliptic PDEs, p. 130-193) | examples and Sage exercises |
| part-ii-01-classical-mechanics | Ch. 4 (coupled oscillations, p. 92ff) | examples and exercises |
| part-ii-03-quantum-mechanics | Ch. 8 (quantum mechanics, p. 194ff): numerical solution of the Schrödinger equation | Sage exercises |
| part-i-08-probability | Ch. 10 (stochastic processes, p. 278ff) | exercises |

## Sections, equations and problems used

Problems of Chs. 3-4 adapted in `part-ii-01-classical-mechanics` (task 12.8):

- Problem 3.2 (anharmonic oscillators, period against amplitude): exm-clb-quartic-scaling, exr-clb-anharmonic-period (Sage).
- Problem 3.3 (restoring force of constant magnitude; the printed period omits the amplitude, T = 4 sqrt(2mA/F0)):
  exm-clb-v-potential, exr-clb-constant-force-period.
- Problem 3.4 (impulsive force): exm-clb-impulse.
- Problems 4.1-4.3 (chains with free, periodic and mixed ends): exm-clb-chain-modes, exr-clb-chain-boundaries (Sage).
- Problem 4.4 (reflection of pulses at fixed and free ends): exm-clb-chain-sound.
- Problem 4.7 (mass impurity and localisation): exr-clb-mass-defect (Sage).
- Problem 4.8 (damped chains): exm-clb-uniform-damping, exr-clb-damped-modes.
- Problem 4.9 (driven chain, resonances and anti-resonances): exr-clb-antiresonance.
- Problem 4.12 (square lattice of masses and springs): exm-clb-square-lattice, exr-clb-square-lattice (Sage).
- Not used (task 8 check): the text of Ch. 4 (Secs. 4.1-4.7: the chain's equations, its numerical integration, Fourier
  and power spectra, continuum limit), which the chapter derives in its own notation, the problems carrying the
  material; Problem 4.5 (a general mode programme, no content beyond exr-clb-chain-boundaries), Problem 4.6 (travelling
  wave in a driven periodic chain, a simulation project) and Problems 4.10-4.11 (input power into a driven chain with
  an impurity, simulation projects beyond the chapter's statements). Problems 3.5-3.7 (nonlinear damped and driven
  oscillators, simulation projects) are not used; Problem 3.1 is used in the differential-equations chapter (below).

Chapter part-i-05-differential-equations (textbook pass, task 6). PDF page = printed page + 15.

- Secs. 3.2-3.3, eqs. (3.8)-(3.25), Figs. 3.2-3.5 (Euler, Euler-Cromer, Verlet and velocity Verlet steps; energy error):
  prp-ts-oscillator-steps, exm-ts-euler-growth, exm-ts-energy-error, exr-ts-frequency (Sage), exr-ts-stability-limit.
- Sec. 3.5, eqs. (3.26)-(3.35), Fig. 3.9 (damped oscillator by Euler-Cromer): exm-ts-damped; Problem 3.1 (complex
  amplitudes): exr-ho-complex-fit.
- Sec. 3.7, eqs. (3.45)-(3.53), Figs. 3.19-3.20 (pendulum period, elliptic integral and its series): exr-ts-pendulum (Sage).
- Sec. 5.2, eqs. (5.15)-(5.24), and Problem 5.3 (heat kernel; images on a finite rod): exr-heat-images (Sage).
- Secs. 5.3-5.4, eqs. (5.30)-(5.49), Figs. 5.4-5.5 (FTCS, its stability threshold, implicit and Crank-Nicolson schemes):
  prp-fdm-stability, exm-fdm-ftcs-threshold, exr-fdm-heat-ftcs (Sage), exr-fdm-crank-nicolson.
- Sec. 5.5, eqs. (5.53)-(5.69), Fig. 5.8, and Problem 5.7 (neutron diffusion, critical length): exr-sep-neutron-critical (Sage).
- Sec. 6.2, eqs. (6.3)-(6.10), Fig. 6.1 (FTFS, FTCS, upwind and Lax schemes, CFL condition): prp-fdm-stability,
  exm-fdm-courant-one, exm-fdm-upwind.
- Sec. 6.4, eqs. (6.26)-(6.30) (leapfrog scheme and its start): prp-fdm-stability, exm-fdm-courant-one,
  exr-fdm-leapfrog-start; eqs. (6.33)-(6.36) (first-order form of the wave equation): exm-wave-first-order.
- Secs. 7.2-7.3, eqs. (7.3)-(7.4), (7.25)-(7.38), Figs. 7.3-7.5 (discrete Laplace equation, Jacobi, Gauss-Seidel, SOR):
  prp-fdm-relaxation, exm-fdm-six-points, exm-fdm-jacobi-rate, exm-fdm-gauss-seidel, exr-fdm-discrete-harmonic,
  exr-fdm-energy, exr-fdm-jacobi (Sage, with eq. (7.29)).
- Errors noted: eq. (3.14) writes a_{n+1/2} = a_n + a_n tau/2 (the second term should be the rate of change of a, or a
  evaluated at the half step); the caption of Fig. 3.8 gives phi_x = 2.33 and phi_x = 3 for omega_x = 2.33 and
  omega_y = 3; in Sec. 3.5 the damped frequency for omega_0 = 3, gamma = 0.5 is sqrt(9 - 0.0625) = 2.990, not 2.958, so
  the computed 2.992 is accurate to the resolution of the period rather than 1% off (exm-ts-damped); Problem 3.1(c)
  has cos(omega_1 t) in place of sin(omega_1 t) in the second term (exr-ho-complex-fit); eq. (5.47) has the
  denominator (2 dx)^2 instead of 2 dx^2 (exr-fdm-heat-ftcs); eq. (6.30) should read u^{-1} = u^1 - 2 g tau, not
  u^0 - 2 g tau (exr-fdm-leapfrog-start); in Sec. 7.2 the matrix of eq. (7.5) is block tridiagonal, not tridiagonal, and
  it is invertible for every ratio dx/dy, contrary to the remark that a special ratio makes its determinant vanish
  (exm-fdm-six-points, exr-fdm-energy); eq. (7.34) is introduced "when dx = dy" but is the case dx != dy; in eq. (6.47)
  the middle case of the stoplight solution is garbled (it is rho_m (1 - x/(v_m t))/2 for -v_m t < x < v_m t).
- Not used: Sec. 3.4 (Lissajous figures), Secs. 3.8-3.9 (driven nonlinear pendulum, non-sinusoidal driving), Sec. 5.1
  (classification, already in the chapter), the Dufort-Frankel scheme of Sec. 5.4, Secs. 6.3 and 6.5 (implicit advection
  schemes, Burgers and traffic equations), Secs. 7.4-7.5 (Poisson equation, multiple Fourier transform), Problems
  5.1-5.2 (already the chapter's spreading Gaussian), 5.4-5.6, 5.8, 6.1-6.9, which ask for programs without new content
  for the chapter. Sec. 3.1 (the harmonic oscillator, already the chapter's running example) and Sec. 3.6 (driven damped
  oscillator, which the chapter treats analytically in its resonance section) are not used either.

Chapter part-i-08-probability (textbook pass, task 6):

- Sec. 10.2, eqs. (10.2)-(10.5), Figs. 10.2-10.4 (particles in a box): exr-rw-ehrenfest (Sage).
- Sec. 10.3, eqs. (10.6)-(10.15) (one- and two-dimensional walks, mean, mean-squared displacement, P(m, N)):
  def-random-walk, exm-rw-moments, exm-rw-distribution, exm-rw-square.
- Problems 10.2-10.4 (P(m, N), its moments and its Gaussian limit): exm-rw-kernel, exr-rw-gaussian (Sage); Problem 10.11
  (triangular lattice): exr-rw-triangular.
- Sec. 10.4.4, Fig. 10.25 (mean first-passage time x0(L - x0)): exr-rw-first-passage (Sage).
- Sec. 10.5, eqs. (10.18)-(10.23) (master equation, diffusion and biased diffusion limits): prp-rw-diffusion-limit,
  exm-rw-kernel, exm-rw-drift.
- Sec. 10.6, eqs. (10.24)-(10.26), and Problem 10.12 (entropy of a walk between reflecting walls): exm-rw-reflecting,
  exr-rw-entropy (Sage).
- Errors noted: in Sec. 10.2 the update rule ("choose a half at random, then move one of its particles") has no drift
  and does not lead to the rate equation (10.3), which is the rule "move a particle chosen at random" (the Ehrenfest
  model); the quoted relative fluctuations (0.088, ..., five values for six N) do not follow the 1/sqrt(N) of the
  binomial equilibrium (exr-rw-ehrenfest); eq. (10.26) lacks the factors 1/2 and imposes P(+-L, n) = 0, which describes
  absorbing rather than reflecting walls (exm-rw-reflecting).
- Not used: Secs. 10.1, 10.4.1-10.4.3 and 10.4.5 (persistent walk, variable step lengths, continuum walk, self-avoiding
  walk), Problems 10.1, 10.5-10.10, 10.13-10.14, which are simulation projects beyond the chapter's statements.

Chapter part-ii-03-quantum-mechanics (task 8 check):

- Not used in this pass: Ch. 8 (numerical Schrödinger equation: Secs. 8.2-8.8, time stepping, expectation values,
  wavepackets, shooting for the finite well and the oscillator, variational method). The chapter's numerical sections
  were written from Izaac-Wang Part III (Chs. 9-11), which covers the same algorithms with the finite-difference
  analysis the chapter uses; Foulaadvand's Ch. 8 would repeat them without new statements.
