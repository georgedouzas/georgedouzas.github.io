# Identity

- arXiv: 0802.0634 (v1, 5 Feb 2008); preprint LPTENS-08/05.
- Title: Lectures on Anomalies. Author: Adel Bilal (LPT, Ecole Normale Superieure, Paris). 108 PDF pages.

# Structure

1. Introduction. 2. Notations for gauge theories. 3. Transformation of the fermion measure and the abelian anomaly; 3.7
relation with the index of the Dirac operator, eqs. (3.39)-(3.47) [read]. 4-7. Anomalies, triangle diagrams, cancellation
(not used). 8. Differential forms and characteristic classes (8.3.1). 9. Wess-Zumino consistency. 10. Relation between
anomalies and index theorems, eqs. (10.51)-(10.54) (Chern character) [read]. 11. Gravitational and mixed anomalies: 11.2,
eqs. (11.14)-(11.17) (Dirac operator with spin connection, A-hat genus) [read]; 11.3.1, eq. (11.25) (mixed index) [read].
12. Ten-dimensional anomaly cancellation (not used). 14. Appendix: explicit two-dimensional illustration of the index theorem,
eqs. (A.1)-(A.8) [read].

# Conventions and mapping to the tutorial

- Hermitian field strength with [D_mu, D_nu] = -i F_mu nu; the mathematical (anti-Hermitian) convention F = -i F_Herm is used in
  (3.47) and (10.53), ch(F) = tr exp(i F / 2 pi). The tutorial's potentials are anti-Hermitian with A = i a, F = i da, and its Chern
  character is written tr exp(-i F / 2 pi) so that the monopole has ch_1 = da / 2 pi (def-chern-character); the overall sign of
  the index formula is fixed in the tutorial by the explicit sphere computation (thm-monopole-index).
- Euclidean chirality gamma_E = i^r gamma^1 ... gamma^{2r}, the same normalisation as the tutorial's gamma_* for Euclidean
  signature.

# Sections and equations used (part-i-13-spinors-dirac-operator)

- Eqs. (3.40), (3.46)-(3.47) (index as a supertrace, index in 2r dimensions): thm-atiyah-singer, exr-spd-supertrace.
- Eq. (10.53) (Chern character): def-chern-character.
- Eqs. (11.14)-(11.17) (Dirac operator with spin connection; A-hat genus (11.15)-(11.16); A-hat has only 4k-forms): def-a-hat-genus,
  def-twisted-dirac-operator, thm-chirality-no-go. Eq. (11.25) (index with gauge and gravitational fields): thm-atiyah-singer.
- Appendix 14, eqs. (A.1)-(A.8) (instanton on the plane, zero modes g(zbar)(1 + z zbar)^{-m/2}, index m): thm-monopole-index,
  exm-spd-plane-versus-sphere, exr-spd-index-density (eqs. (A.1)-(A.2)).
