# 300 Problems in Special and General Relativity: With Complete Solutions

Mattias Blennow and Tommy Ohlsson, Cambridge University Press, 2022. Local copy: `references/textbooks/300 Problems in Special and General Relativity.pdf` (git-ignored). Textbook (design D32).

## Chapters it serves

Mapped from the book's table of contents (chapter titles and page numbers of the PDF). The sections, equations
and problem numbers actually used are added below when the chapter that uses them is written or revised.

| Chapter of this book | Sections of the textbook | Use |
|---|---|---|
| part-ii-09-general-relativity | Ch. 2 (general relativity problems, p. 53ff) with solutions in Ch. 3 (p. 87ff): geodesics, Christoffel symbols, curvature of explicit metrics, Schwarzschild orbits, cosmology | exercises |
| part-ii-04-classical-field-theory | Ch. 1 (special relativity, p. 24ff): Lorentz transformations, four-vectors, relativistic kinematics | exercises |
| part-ii-08b-special-relativity | Ch. 1, Secs. 1.1-1.9 (problems 1.1-1.150, p. 14ff) with solutions in Sec. 3.1 (p. 77ff) | main source of examples and exercises |
| part-i-09-differential-geometry | Ch. 2 problems on tensors and curvature of explicit metrics | exercises |

## Sections, equations and problems used

Chapter 2 (general relativity), adapted in `part-ii-09-general-relativity` (task 12.8):

| Problem | Block | Use |
|---|---|---|
| 2.72 | `exm-grb-kepler-exact` | Kepler's third law exact in coordinate time, proper period |
| 2.75 | `exr-grb-epicycles` | radial and orbital periods of near-circular orbits (Sage) |
| 2.76 | `exm-grb-capture` | critical impact parameter as capture cross-section |
| 2.77 | `exm-grb-fronsdal` | Fronsdal embedding in six-dimensional Minkowski space |
| 2.79 | `exm-grb-ds2-global` | light on the two-dimensional de Sitter hyperboloid |
| 2.83 | `exm-grb-ds-straight` | straight comoving geodesics and arrival time in flat-sliced de Sitter |
| 2.86 | `exm-grb-ads-global`, `exr-grb-ads-null` | global AdS4 coordinates; null geodesics straight in the embedding |
| 2.91, 2.96 | `exm-grb-max-proper-time` | longest proper time to the singularity |
| 2.94, 2.100 | `exm-grb-static-acceleration` | proper acceleration of static observers |
| 2.99, 2.100 | `exr-grb-orbit-vs-static` | orbiting and static clocks, relative speed (Sage) |
| 2.107, 2.108 | `exr-grb-kruskal-trapped` | no escape from the region u, v > 0 |
| 2.110 | `exr-grb-frame-curvature` | curvature invariant at r = 0 (Sage). The problem statement (eq. 2.117) prints the Kretschmann scalar as 3 r*^2/r^6; the correct value is 12 r*^2/r^6 = 48 G^2M^2/r^6 (only the r^-6 behaviour matters for the question asked) |
| 2.112 | `exr-grb-linearised-gauge` | linearised Ricci tensor, gauge transformation, Lorenz gauge |
| 2.113 | `exm-grb-pound-rebka` | gravitational redshift in the Earth's field |
| 2.115 | `exr-grb-rotating-frame` | Christoffel symbols of a rotating frame as centrifugal and Coriolis forces |
| 2.125 | `exm-grb-solar-lines` | solar spectral lines seen from the Earth |
| 2.126 | `exr-grb-rocket-signal` | gravitational versus Doppler shift (Sage) |
| 2.133 | `exm-grb-gw-polarisations`, `exm-grb-gw-ruler` | TT gauge counting; proper distance in a plus-polarised wave |
| 2.137 | `exm-grb-milne` | linearly expanding (Milne) universe |
| 2.139 | `exr-grb-ds2-redshift` | redshift in two-dimensional de Sitter space (Sage) |
| 2.140 | `exr-grb-ds2-wave` | wave operator on two-dimensional de Sitter space (Sage) |
| 2.143 | `exm-grb-inflation-flatness`, `exr-grb-flatness` | flatness problem and inflation |
| 2.144 | `exm-grb-radiation-equality` | matter-radiation equality redshift |
| 2.146 | `exr-grb-circle-universe` | travelling around a closed 1+1 universe |
| 2.147 | `exr-grb-equality-time` | time since matter-radiation equality (Sage) |
| 2.148 | `exr-grb-flatness` | condition w < -1/3 for the curvature to fade |

Chapter 1 (special relativity), adapted in `part-ii-08b-special-relativity` (task 12.9). Blennow and Ohlsson use the
signature (+, -, -, -) and $F^{\mu\nu}$ with $F^{i0} = E^i$, as the chapter does; they keep $c$, which the chapter sets to $1$ except in numbers.
Problem statements were adapted, and several were given new numbers; the solutions were checked against Sec. 3.1.

| Problem | Block | Use |
|---|---|---|
| 1.1 b | `exr-sr-which-velocity` | causal character, which vectors are four-velocities |
| 1.2 | `prp-sr-reverse-inequalities`, `exr-sr-null-decomposition` | reverse inequalities; spacelike vector as a difference of null vectors |
| 1.3 | `exr-sr-relative-gamma` | Lorentz factor of a relative motion (Sage) |
| 1.4 | `exm-sr-photon-rest-frame` | no rest frame for a photon, one for two photons |
| 1.6 | `exr-sr-inclined-rod` | contraction of an inclined rod ($v = \sqrt{2/3}$, $60°$) |
| 1.7 | `exm-sr-muons` | cosmic-ray muons |
| 1.8 | `exr-sr-train-length`, `exm-sr-radar` | four measurements of a train's length; radar (d) uses the solution's single-pulse reading |
| 1.9 | `exm-sr-cruiser` | speed from a simultaneity mismatch, $v = c^2\tau/L$ |
| 1.10 | `exr-sr-two-lamps` | distance and time between lamps |
| 1.13 | `exr-sr-interval-distance` | distance from the invariant interval (Sage) |
| 1.14 | `exr-sr-muon-energy` | muon energy needed to reach the ground |
| 1.15 | `exr-sr-muon-ring` | muons in a ring (Sage) |
| 1.16 | `exr-sr-triangle` | moving 3-4-5 triangle (Sage) |
| 1.17 | `exr-sr-pole-marks` | marks on a moving pole seen at three angles (Sage) |
| 1.18 | `exr-sr-bell` | Bell's spaceships (Sage) |
| 1.20 | `exr-sr-two-lengths` | rockets of lengths $L$, $2L$; reformulated as an overtaking with timelike $A$, $B$ |
| 1.23 | `exm-sr-lamps-approach` | apparent separation of approaching lamps |
| 1.24 | `exr-sr-light-speed` | the boost keeps the speed of light |
| 1.25 | `exm-sr-lanterns` | lanterns on a train |
| 1.27 | `exm-sr-starry-sky` | the sky from a fast ship |
| 1.31 | `exr-sr-twisted-cylinder` | twisted rotating cylinder (Sage) |
| 1.32 | `exr-sr-light-sweep` | superluminal illumination sweep |
| 1.34 | `exr-sr-perpendicular-chain` | perpendicular velocities in a chain |
| 1.35 | `exm-sr-vector-from-spinors`, `exr-sr-null-spinor` | null vector as a spinor (Sage) |
| 1.37 | `exr-sr-two-rockets` | signals to two rockets (Sage) |
| 1.38 b | `exm-sr-worldline-limited`, `exr-sr-proper-time-limited`, `exr-sr-four-acceleration` | accelerated worldline, proper time, four-acceleration (Sage) |
| 1.41 | `exm-sr-andromeda` | trip to Andromeda at $1g$ |
| 1.42 | `exm-sr-rocket-equation`, `exr-sr-rocket-derivation` | relativistic rocket equation |
| 1.43 | `exr-sr-message-friend` | direction of a message to a friend |
| 1.44 | `exm-sr-u-product`, `exr-sr-gamma-other-frame` | Lorentz factors from four-velocities |
| 1.45 | `exr-sr-u-dot-a` | $u\cdot a = 0$ |
| 1.46 | `exr-sr-rescue` | rescue in deep space (Sage) |
| 1.47 | `exr-sr-police` | relativistic pursuit (Sage) |
| 1.48 | `exm-sr-rindler-metric`, `exr-sr-rindler-photon`, `exr-sr-rindler-sideways` | Rindler metric, photon components, sideways motion (Sage) |
| 1.49 | `exr-sr-hyperbola-invariant` | hyperbolic worldline in another frame |
| 1.50 | `exr-sr-signal-to-rocket` | signal to an accelerating rocket, horizon (Sage) |
| 1.51 | `exm-sr-u-circle`, `exr-sr-accelerator-orbit` | circular motion, proper acceleration (Sage) |
| 1.52 | `exr-sr-force-along-u` | four-force along a fixed vector |
| 1.53 | `exm-sr-spatial-force` | four-force without time component |
| 1.54 | `exm-sr-acc-transform` | acceleration in a moving frame |
| 1.55 | `exm-sr-energy-observer`, `exr-sr-energy-moving-observer` | energy measured by an observer |
| 1.56 | `exr-sr-null-force` | constant null four-force (Sage) |
| 1.57 | `exr-sr-oblique-particle` | particle hitting a moving observer (Sage) |
| 1.58 | `exr-sr-coordinate-acceleration` | constant coordinate acceleration |
| 1.59 | `exm-sr-fizeau` | Fizeau's drag coefficient |
| 1.61 | `exm-sr-quasar` | quasar 3C 9 as a Doppler shift |
| 1.62 | `exm-sr-wave-along-x`, `exr-sr-doppler-rapidity` | wave four-vector, Doppler factor in rapidity |
| 1.63 | `exm-sr-grb` | stretched gamma-ray bursts |
| 1.64 | `exm-sr-half-plus-half`, `exr-sr-ruby` | relative speed $0.8c$; ruby laser blueshift |
| 1.65 | `exr-sr-doppler-unshifted` | direction of no Doppler shift (Sage) |
| 1.66 | `exr-sr-disk-clocks` | clocks on a rotating disk |
| 1.68 | `exr-sr-moving-mirror` | moving mirror (Sage) |
| 1.69 | `exr-sr-moving-medium` | refraction without bending; the condition $n^2 = (1 - 2w\cos\theta' + w^2)/w^2$ is derived in the chapter (Sage) |
| 1.70 | `exm-sr-wave-medium`, `exr-sr-doppler-medium` | waves in a medium |
| 1.71 | `exr-sr-reflected-signal` | reflected signal to an accelerated observer |
| 1.72 | `exr-sr-accelerator-rapidity` | electron after one megavolt (Sage) |
| 1.73 | `exm-sr-no-single-photon` | no annihilation into one photon |
| 1.74 | `exm-sr-daughter-frame` | one daughter seen from the other |
| 1.77 | `exm-sr-merge` | two particles merge |
| 1.78 | `exr-sr-muon-right-angle` | muon emitted at right angles |
| 1.80 | `exm-sr-pion-decay` | $\pi^+ \to \mu^+\nu_\mu$ |
| 1.82 | `exr-sr-resonance` | resonance in two muons (Sage) |
| 1.84 | `exr-sr-annihilation-angle` | annihilation in flight |
| 1.88 | `exm-sr-t-elastic`, `exr-sr-recoil` | momentum transfer, recoil energy |
| 1.89 | `exr-sr-delta-resonance` | forming the $\Delta(1232)$ |
| 1.93 | `exr-sr-deuteron-photon` | neutron capture |
| 1.94 | `exr-sr-photoionisation` | photoionisation threshold |
| 1.95 | `exm-sr-z-burst` | $Z$ production on relic neutrinos |
| 1.96 b | `exr-sr-compton-cm` | Compton scattering in the centre-of-momentum frame |
| 1.97 | `exm-sr-compton-electron` | Compton recoil electron |
| 1.98 | `exr-sr-inverse-compton` | inverse Compton scattering (Sage) |
| 1.99 | `exr-sr-muon-decay-max` | largest neutrino energy in muon decay |
| 1.102 | `exm-sr-two-photons` | diphoton invariant mass |
| 1.103 | `exm-sr-antiproton` | antiproton threshold |
| 1.104 | `exm-sr-pion-production` | $\pi^-p \to \pi^+\pi^-n$ threshold |
| 1.105 | `exr-sr-celsius` | CELSIUS reaction (Sage) |
| 1.107 | `exr-sr-moller-threshold` | pair production in $e^-e^-$ |
| 1.108 | `exr-sr-kaon-threshold` | kaon thresholds (Sage) |
| 1.110 | `exr-sr-neutrino-delay` | supernova neutrino delay (Sage) |
| 1.111 | `exr-sr-linac-time` | electron in a linear accelerator (Sage) |
| 1.112 | `exr-sr-box-invariant` | invariance of $\Box A^\mu = 0$ |
| 1.114, 1.115 | `exm-sr-field-electric-boost` | boosted electric field |
| 1.117, 1.132 | `exm-sr-moving-charge`, `exm-sr-invariants-moving-charge`, `exr-sr-moving-charge-potential` | field of a moving charge (Sage) |
| 1.118 | `exm-sr-constant-e` | constant electric field |
| 1.119 | `exm-sr-wire` | current-carrying wire |
| 1.121 | `exm-sr-plane-wave-null` | plane wave invariants |
| 1.123, 1.133 a | `exm-sr-constant-b` | constant magnetic field, momentum from the radius |
| 1.124, 1.125 | `exr-sr-unknown-components` | unknown field components from the invariants (Sage) |
| 1.128 | `exr-sr-muon-deflection` | muon in the Earth's field (Sage) |
| 1.129 | `exr-sr-b-vanishes` | frame without magnetic field |
| 1.130 | `exr-sr-b-boost` | boosted magnetic field (Sage) |
| 1.131 | `exr-sr-slac` | electron at SLAC (Sage) |
| 1.133 b | `exr-sr-missing-mass` | missing mass in a bubble chamber (Sage; momenta generated from a decay into a neutron) |
| 1.135 | `exr-sr-angle-invariant` | angle between E and B |
| 1.141 | `exr-sr-dipole-region` | where the electric field can vanish |
| 1.143 | `exm-sr-parallel-charges` | force between parallel moving charges |
| 1.145 | `exr-sr-dust` | ship entering a dust cloud |
| 1.147, 1.148 | `exr-sr-fluid-boost` | perfect fluid seen from a moving frame (Sage) |
| 1.150 | `exr-sr-quadratic-lagrangian`, `exr-sr-minimal-coupling-force` | quadratic Lagrangian and minimal coupling |

Section-level citations in the chapter's part introductions (task 8 check): Secs. 1.1-1.4 (postulates, Lorentz
transformations, Minkowski spacetime), before def-sr-inertial-frame and def-sr-minkowski-spacetime; Secs. 1.4 and 1.6
(kinematics), before def-sr-four-velocity; Sec. 1.5 (optics), before def-sr-wave-vector; Secs. 1.4, 1.6 and 1.9 (dynamics),
before def-sr-four-force; Sec. 1.7 (electrodynamics), before prp-sr-field-transformation.

Ch. 1, mapped also to `part-ii-04-classical-field-theory`: not used there (task 8 check). Its special-relativity problems
were placed in `part-ii-08b-special-relativity` (above), which the classical-field-theory chapter cites for Lorentz
transformations and four-vectors instead of repeating them. The problems of Ch. 1 not listed above (about
forty-five of the 150, among them 1.11-1.12, 1.19, 1.21-1.22, 1.26, 1.28-1.30, 1.33, 1.36, 1.39-1.40, 1.75-1.92 in part, 1.134-1.149 in
part) are not used in this pass: every statement of the chapter already carries its examples and exercises from the
problems above, and these would add variants of the same computations.

The sign of the second invariant: Blennow and Ohlsson, eq. (0.27), quote $-8cB\cdot E$ for their placement of the indices and their sign of
$\epsilon$; with lower-index $F$ and $\epsilon_{0123} = 1$ the chapter finds $\epsilon^{\mu\nu\rho\sigma}F_{\mu\nu}F_{\rho\sigma} = +8\vec E\cdot\vec B$ (exr-sr-invariants-components, Sage).

Chapter 2 (differential geometry problems), adapted in `part-i-09-differential-geometry` (textbook pass, task 2):

| Problem | Block | Use |
|---|---|---|
| 2.3 | `exr-christoffel-hyperbolic-coordinates` | plane in coordinates $x = se^t$, $y = se^{-t}$ and $y = v^2$ (Sage) |
| 2.20 | `exr-sff-hyperboloid` | hyperboloid of one sheet in Euclidean space, here through its principal curvatures |
| 2.41 | `exr-sff-graph`, `exr-sff-paraboloid` | Riemann tensor of the paraboloid, intrinsically and by Gauss's equation (Sage) |
| 2.42 | `exr-gauss-cone-excess` | $r^2(dr^2 + r^2d\varphi^2)$ is a cone of total angle $4\pi$, $C^2 = 8\pi A$ |
| 2.46 | `exr-sff-catenoid-helicoid` | the metric $d\rho^2 + (a^2 + \rho^2)d\varphi^2$ realised by the catenoid and the helicoid (Sage) |
| 2.48 | `exr-einstein-de-sitter` | two-dimensional de Sitter space, $R_{ij} = -g_{ij}$ in the chapter's conventions |
