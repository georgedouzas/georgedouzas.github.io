# Identity

- arXiv: gr-qc/9505009 (v1, 9 May 1995).
- Title: On the eigenfunctions of the Dirac operator on spheres and real hyperbolic spaces.
- Authors: Roberto Camporesi (Politecnico di Torino), Atsushi Higuchi (Bern). Journal version J. Geom. Phys. 20 (1996) 1
  (external knowledge, not stated in the file). 45 PDF pages.

# Structure

1. Introduction. 2. Gamma matrices and spinor representations of Spin(N) (eqs. (2.1)-(2.23)). 3. Spherical modes of the
Dirac operator: 3.1 S^N (geodesic polar coordinates, eqs. (3.1)-(3.9); N even, eigenfunctions and degeneracy (3.41)-(3.45);
N odd) [read]. 4. Real hyperbolic spaces (not used). 5. Heat kernel and spectrum from representation theory on symmetric
spaces U/K: Frobenius reciprocity (5.3), Laplacian as difference of Casimirs (5.6), Lichnerowicz (5.35)-(5.36), Casimirs of
Spin(N+1) and Spin(N) (5.41)-(5.48) [read]. 6. Conclusions.

# Conventions and mapping to the tutorial

- Euclidean gamma matrices with {Gamma_a, Gamma_b} = 2 delta_ab and the anti-Hermitian Dirac operator nabla-slash =
  Gamma^a nabla_a, eigenvalues +-i(n + N/2). The tutorial's D = i gamma^a nabla_a has eigenvalues +-(n + N/2).
- Spin connection nabla_a psi = e_a psi - (1/2) omega_abc Sigma^bc psi (3.9), consistent with the tutorial's
  eq-spd-spin-connection up to the normalisation of Sigma.

# Sections and equations used (part-i-13-spinors-dirac-operator)

- Eq. (3.9): def-dirac-operator. Eq. (3.44) (degeneracy D_N(n) = 2^{N/2} (N+n-1)!/(n!(N-1)!)): thm-sphere-dirac-spectrum,
  prp-s3-dirac, figure sd-sphere-spectrum.
- Eq. (5.3) and footnote 3 (homogeneous spinor bundles, Frobenius reciprocity, spin lift of the isotropy representation):
  def-homogeneous-spinor-bundle, thm-frobenius-reciprocity, exm-spd-s2-homogeneous; remark after (5.3) on multiplicities
  xi_lambda: exm-spd-multiplicity-one.
- Eq. (5.6) (Laplacian = C_2(lambda) - C_2(tau)) and (5.35)-(5.37) (Lichnerowicz, -nabla-slash^2 = -nabla^a nabla_a + R/4):
  thm-lichnerowicz, thm-lichnerowicz-vanishing, thm-parthasarathy, exr-spd-casimir-laplacian.
- Eqs. (5.41)-(5.48) (Freudenthal Casimirs, C_2(lambda_n) = (n + N/2)^2 - N(N-1)/8, C_2(tau_+-) = N(N-1)/8): exm-spd-sn-representations,
  exm-spd-parthasarathy-sn, exr-spd-spinor-casimir, exm-spd-sn-frobenius.
