# Advanced Newtonian Rigid Dynamics

Rajnikant Sinha, Springer, 2023. Local copy: `references/textbooks/Sinha - 2023 - Advanced Newtonian Rigid Dynamics.pdf` (git-ignored). Textbook (design D32).

## Chapters it serves

Mapped from the book's table of contents (chapter titles and page numbers of the PDF). The sections, equations
and problem numbers actually used are added below when the chapter that uses them is written or revised.

| Chapter of this book | Sections of the textbook | Use |
|---|---|---|
| part-ii-01-classical-mechanics | Ch. 1 (3-D kinematics of rigid bodies, p. 10ff), Ch. 3 (momental ellipsoid, p. 258ff), Ch. 7 (motion in three dimensions, p. 812ff) | examples and exercises |
| part-i-11-lie-groups-algebras | Ch. 1 (rotations and Euler angles) | examples |

## Sections, equations and problems used

Worked problems ("Example" subsections) adapted in `part-ii-01-classical-mechanics` (task 12.8), cited by subsection,
`[@sinha2023rigid, Sec. 2.4.6]`:

- Sec. 2.4.1 (compound pendulum, equivalent simple pendulum, centre of oscillation): exm-clb-compound-pendulum.
- Secs. 2.4.2-2.4.3 (equivalent lengths: circular wire, hemisphere, cube): exr-clb-equivalent-lengths (Sage).
- Sec. 2.4.6 (three particles on a rod, least period): exr-clb-three-particles.
- Sec. 2.4.8 (cone swinging about its vertex): exm-clb-cone-pendulum.
- Sec. 2.4.11 (axes of equal period, Kater's pendulum): exr-clb-kater.
- Sec. 3.4.3 (principal axes of a rectangle at a corner): exm-clb-rectangle-axes.
- Sec. 3.4.12 (principal axes at a corner of a cube): exm-clb-cube-corner.
- Sec. 7.1.5 (rod sliding on two perpendicular lines, instantaneous centre and centrodes): exm-clb-sliding-ladder, exr-clb-ladder-dynamics.
- Secs. 7.2.11 and 7.2.13 (sphere rolling on an incline: five-sevenths of the force, friction 2/7 tan alpha): exm-clb-rolling-sphere, exr-clb-rolling-race.
- Sec. 8.2.1 (contracting Earth and the length of the day): exm-clb-earth-contraction.
- Sec. 8.2.2 (insect walking on a turntable): exm-clb-insect-disc.
- Sec. 8.2.4 (insect walking on a free ring): exr-clb-insect-ring.
- Sec. 8.2.12 (point of a spinning plate suddenly fixed, omega/3): exr-clb-plate-fixed-point.
- Sec. 8.2.18 (cube caught by an edge, sqrt(3) omega/12): exr-clb-cube-edge (Sage).
- Chs. 2 and 8 (compound pendulum, impulsive motion), not in the mapping table, are used as listed. Ch. 1 (kinematics
  of rigid bodies), mapped here, is used in part-i-11-lie-groups-algebras instead (below), where angular velocity and Euler's
  kinematic equations are stated; the classical-mechanics chapter does not repeat them. Not used (task 8
  check): the remaining worked problems of Chs. 3 and 7 (moments of inertia of further bodies, motion of tops and
  rolling bodies in three dimensions), which repeat the computations of the examples above without new content.

### part-i-11-lie-groups-algebras (textbook pass, task 4)

Ch. 1, translated to this book's z-x-z Euler angles $e^{\varphi L_3}e^{\theta L_1}e^{\psi L_3}$ (Sinha's z-y-z angles
$(\psi, \theta, \varphi)$ correspond to ours with $\varphi = \psi_S + \pi/2$, $\psi = \varphi_S - \pi/2$):

- Sec. 1.1.4 (composition of two finite rotations about concurrent axes, spherical construction): exm-lg-comp-order.
- Sec. 1.1.5 (Hamilton's theorem): exr-lg-comp-hamilton (Sage).
- Sec. 1.1.6 (rotations about parallel axes; equal and opposite angles give a translation): exr-lg-comp-parallel-axes.
- Secs. 1.1.28-1.1.29 (rotation matrices, Eulerian angles in the z-y-z convention): def-lg-angular-velocity, exm-lg-ek-pole.
- Secs. 1.1.30-1.1.31 (Euler's geometrical equations in body and space axes): prp-lg-euler-kinematic, exm-lg-ek-pole,
  exr-lg-ek-check (Sage).
- Sec. 1.1.33 (angle swept by a line moving relative to rotating axes): exr-lg-angvel-moving-line.
