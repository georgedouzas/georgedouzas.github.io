# Identity

- arXiv: 1102.2001 (v2, 11 June 2011).
- Title: Elementary analysis of the special relativistic combination of velocities, Wigner rotation, and Thomas precession.
- Authors: Kane O'Donnell and Matt Visser (Victoria University of Wellington).
- Published: European Journal of Physics 32 (2011) 1033-1047. 22 pages.

# Structure

1. Introduction (p. 3). 2. Qualitative introduction: combination of velocities, Wigner rotation, Thomas precession
(pp. 4-7, Figs. 1-3). 3. Elementary level: parallel and perpendicular velocities, eqs. (2)-(8); general velocities,
eqs. (9)-(26), Stapp's formula eq. (21). 4. Boost-matrix formulation: eqs. (27)-(49), composition $L_{21} = B_2B_1 = RB_{21}$,
the reversed order eqs. (38)-(40), $R_3\vec v_{21} = \vec v_{12}$ eq. (46). 5. Spinor formulation: boosts $\cosh(\xi/2) + \sinh(\xi/2)\,\hat n\cdot\vec\sigma$
and rotations eqs. (50)-(53), the coefficient equations (54)-(57), results (58)-(59), trace argument (60)-(64).
6. Summary of formulae, eqs. (65)-(78). 7. Further reading.

# Conventions

- $c = 1$; boosts are passive ($\vec x' = B\vec x$ maps to the frame moving with $\vec v$), so their $B_2B_1$ is our
  $B(-\vec v_2)B(-\vec v_1)$. The chapter works with active boosts $B(\vec v)$ (eq-sr-boost-matrix) and decomposes
  $B(\vec v_2)B(\vec v_1) = B(\vec u)W$; with this convention the rotation is by $+\Omega$ about $\hat n_1\times\hat n_2$, checked numerically
  (sage/exercises/exr-sr-wigner-random.sage).
- Thomas precession: eq. (26) gives $\frac{\gamma}{1 + \gamma}\vec v\times\vec a$ "in mission control's frame", eq. (25) and footnote 4 discuss the
  $\frac{\gamma^2}{1 + \gamma}$ form "as viewed from Alice's frame". The chapter defines the precession as the rotation of the Fermi-Walker
  frame relative to the pure-boost frame per laboratory time, which is $\frac{\gamma^2}{\gamma + 1}\vec a\times\vec v$, and checks it against an
  integration of the Fermi-Walker equation (sage/exercises/exr-sr-fermi-walker.sage, angle $2\pi(\gamma - 1)$ per orbit).

# Sections, equations and problems used (part-ii-08b-special-relativity, task 12.9)

- Secs. 2-4 (combination of velocities, composite velocity eqs. (13)/(36)): def-sr-wigner-rotation,
  exr-sr-composite-velocity, exm-sr-wigner-perpendicular.
- Eqs. (38)-(40) and (46) (reversed order, angle between $\vec v_{12}$ and $\vec v_{21}$): exm-sr-wigner-reverse,
  exr-sr-wigner-reversed.
- Sec. 5, eqs. (50)-(59) (spinor derivation): thm-sr-wigner-angle; our half-angle form
  $\tan(\Omega/2) = \sin\theta/(\coth\frac{\xi_1}{2}\coth\frac{\xi_2}{2} + \cos\theta)$ is derived from their coefficient equations (55) and (57), and the
  cosine form eq. (58)/(70) is reproduced.
- Eq. (21)/(69) (Stapp's formula): exr-sr-stapp.
- Eqs. (7)-(8), (24)-(26) (Thomas precession from infinitesimal Wigner rotations): prp-sr-thomas.
