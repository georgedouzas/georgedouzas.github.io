# Identity

- arXiv: hep-th/0005172 (v2, 26 May 2000). Preprint number CTP TAMU-xx/00, dated May 2000.
- Title: "Kaluza-Klein Consistency, Killing Vectors and Kähler Spaces".
- Authors: P. Hoxha, R. R. Martinez-Acosta, C. N. Pope (Center for Theoretical Physics, Texas A&M University).
- Year: 2000. Journal venue: not stated in the PDF (preprint form only).
- PDF file: `references/papers/hep-th_0005172.pdf`, 44 PDF pages. Printed page number = PDF page − 1 throughout (title page unnumbered, Contents is printed p. 1).

# Structure

1. Introduction — printed p. 2 (PDF 3) to p. 6 (PDF 7).
2. Consistency conditions on Killing vectors in Kaluza-Klein reductions — printed p. 6 (PDF 7) to p. 11 (PDF 12). The section that matters most for the tutorial: metric ansatz with Killing vectors, Ricci tensor, the necessary consistency condition.
3. Construction of Killing vectors on the internal space — printed p. 11 (PDF 12).
   - 3.1 Killing vectors on U(1) bundles — p. 11 (PDF 12).
   - 3.2 Killing vectors on Kähler spaces and their U(1) bundle spaces — p. 13 (PDF 14).
   - 3.3 Killing vectors on a product of Kähler spaces and their U(1) bundles — p. 16 (PDF 17).
4. Products of CP^n spaces, and their U(1) bundles — p. 20 (PDF 21).
   - 4.1 Geometry of CP^n, and its Killing vectors — p. 20 (PDF 21).
   - 4.2 Killing spinors on Q^{q_1...q_N}_{n_1...n_N} spaces — p. 23 (PDF 24).
5. Consistency condition for Kaluza-Klein reductions — p. 25 (PDF 26) to p. 34 (PDF 35). Contains the S^n proof (eqs. 5.5–5.13) and the no-go for non-abelian Killing vectors on the Q spaces.
6. Conclusions — p. 35 (PDF 36).
- Appendix A: An iterative construction of CP^n — p. 36 (PDF 37).
- Appendix B: Inequalities on CP^n — p. 37 (PDF 38) to p. 40 (PDF 41).
- References [1]–[36] — p. 40 (PDF 41) to p. 43 (PDF 44).

# Conventions

- Signature: the spacetime vielbein block in (2.3) is `e^α e^β η_{αβ}`; the internal block is `δ_{ab}` (positive definite). η is not written out, but the AdS_5 background satisfies `R_{μν} = −4m² g_{μν}` with the internal sphere `R_{mn} = +4m² g_{mn}` (2.2, p. 7/PDF 8), which is the mostly-plus assignment (AdS has negative Ricci, the sphere positive).
- "Cosmological constant" convention (footnote 4, p. 8/PDF 9): Λ is defined by `R_{ab} = Λ g_{ab}` on an Einstein space. A round sphere S^n has `R_{mn} = (n−1) m² g_{mn}` (p. 10/PDF 11); unit CP^n has Λ = 2(n+1) (4.10, p. 21/PDF 22).
- Riemann sign: never stated explicitly, but fixed by the identities they use: `∇_a ∇_c K_d = R^e{}_{acd} K_e` for a Killing vector (p. 14/PDF 15, after 3.16; also footnote 8, p. 30/PDF 31, as `∇_a ∇_b K_c = R^d{}_{abc} K_d`), `□K_a + R_{ab} K^b = 0` (3.11), and the unit CP^n Riemann tensor `R_{abcd} = δ_{ac}δ_{bd} − δ_{ad}δ_{bc} + J_{ac}J_{bd} − J_{ad}J_{bc} + 2J_{ab}J_{cd}` (4.22, p. 23/PDF 24). Contracting (4.22) as `R_{bd} = R^a{}_{bad}` gives `(D+2)δ_{bd} = 2(n+1)δ_{bd}`, matching (4.10). So: Riemann is positive on spheres (`R_{abcd} = δ_{ac}δ_{bd} − δ_{ad}δ_{bc}` for the unit sphere), and Ricci is the contraction of the first and third indices.
- Kähler identity used: `R_{abcd} = J_c{}^e J_d{}^f R_{abef}` (p. 14/PDF 15). Kähler form normalised so that `J_{12} = J_{34} = ... = +1` in an orthonormal basis (4.32).
- Hats: in section 2, hats denote 10-dimensional quantities; from section 3.1 onward hats denote the total U(1)-bundle space and unhatted the base (stated explicitly p. 11/PDF 12). Indices: α, β spacetime (d = 5); a, b internal orthonormal; m, n internal coordinate; I, J, K adjoint of the isometry group G.
- The KK metric ansatz with Killing vectors, eq. (2.3), p. 8 (PDF 9):

  `dŝ² = e^α e^β η_{αβ} + (e^a − K^{Ia} A^I)(e^b − K^{Jb} A^J) δ_{ab}`

  with `e^α = e^α(x)` the d = 5 spacetime vielbein, `e^a = e^a(y)` the internal vielbein, `K^{Ia} = K^{Ia}(y) = K^{Im}(y) e_m{}^a(y)` the orthonormal-frame components of the Killing vectors generating G, and `A^I = A^I(x) = e^α(x) A^I_α(x)` the Yang-Mills potentials. The gauge field enters as `e^a − K^{Ia} A^I`, i.e. `dy^m − K^{Im} A^I_μ dx^μ` in coordinate form (the `K^m_i A^i_μ` form the tutorial uses, with a minus sign). There are NO warp factors and NO scalar/breathing modes in (2.3): the authors state this is the "leading-order linearised" ansatz for the gauge bosons only (p. 8/PDF 9), sufficient because they only prove inconsistency (p. 7/PDF 8 and footnote 3).
- Killing-vector algebra, eq. (2.4), p. 8: `[K^I, K^J] = f^{IJ}{}_K K^K`, with `f^{IJ}{}_K` the structure constants of G. No separate gauge coupling constant g appears anywhere; it is absorbed into the normalisation of the K^I (the overall normalisation of the Killing vectors is fixed only later, by the requirement `Y^{IJ} = β δ^{IJ}`, (2.12)).
- Field strength, eq. (2.5), p. 8: `F^I = dA^I + ½ f^{IJK} A^J ∧ A^K`, `F^I = ½ F^I_{αβ} e^α ∧ e^β`.
- Gauge-covariant derivative, eq. (2.7), p. 8: `D_γ F^I{}_{αβ} = ∇_γ F^I{}_{αβ} + f^{IJK} A^J{}_γ F^K{}_{αβ}`.
- They do not derive the non-abelian gauge transformation from internal diffeomorphisms; gauge invariance is invoked only as the reason why the linearised ansatz suffices (p. 7/PDF 8).

# Equations the tutorial uses

**(a) General ansatz on M × K with Killing vectors.** Eq. (2.3), p. 8 (PDF 9), as quoted in Conventions. Context: type IIB on AdS_5 × M_5 in the Freund–Rubin background `Ĝ_5 = 4m ε_5`, with equations of motion (2.1), p. 7:

  `R̂_{MN} = (1/96) Ĥ_{MPQRS} Ĥ_N{}^{PQRS},   dĤ_5 = d∗Ĥ_5 = 0`, `Ĥ_5 = Ĝ_5 + ∗̂Ĝ_5`,

and background (2.2): `R_{μν} = −4m² g_{μν}`, `R_{mn} = 4m² g_{mn}`. The gauge fields also enter the 5-form ansatz, (2.9), p. 9: `Ĝ_5 = 4m ε_5 − (1/m) ∗F^I ∧ dK^I` (K^I here is the Killing 1-form on M_5).

The Ricci tensor of (2.3) in the orthonormal basis ê^A, eq. (2.6), p. 8:

  `R̂_{αβ} = R_{αβ} − ½ K^{Ia} K^J{}_a F^I{}_{αγ} F^J{}_β{}^γ`
  `R̂_{ab} = R_{ab} + ¼ K^I{}_a K^J{}_b F^I{}_{αβ} F^{Jαβ}`
  `R̂_{αb} = R̂_{bα} = −½ K^I{}_b (D_β F^I{}_α{}^β)`

and the curvature scalar, eq. (2.8), p. 8:

  `R̂ = R_{(5)} + R_{(M)} − ¼ K^I{}_a K^{Ja} F^I{}_{αβ} F^{Jαβ}`,

where `R_{(5)}`, `R_{(M)}` are the spacetime and internal curvature scalars. These are exact for the metric (2.3) (no approximation is made in computing them; the "linearisation" refers to the field content of the ansatz). Eq. (2.8) is directly comparable with a Sage computation of the reduced curvature scalar on M_4 × S^2 at zero scalar fields: the coefficient of the Yang-Mills term is `−¼ (K^I · K^J) F^I_{αβ} F^{J αβ}`, with the y-dependent "metric on the gauge indices" `K^I_a K^{Ja}`.

**(b) Reduced equations of motion.** The paper derives no reduced action and no scalar sector. It derives the five-dimensional spacetime components of the ten-dimensional Einstein equation, eq. (2.10), p. 9 (PDF 10):

  `R_{μν} − ½ g_{μν}(R_{(5)} + R_{(M)}) = ½ (F^I{}_{μρ} F^{Jρ}{}_ν − ¼ g_{μν} F^I{}_{σρ} F^{Jσρ}) Y^{IJ}`,

with eq. (2.11):

  `Y^{IJ} = Y(K^I, K^J) ≡ K^I{}_m K^{Jm} + (1/(2m²)) ∇_m K^I{}_n ∇^m K^{Jn}`.

If `Y^{IJ} = β δ^{IJ}` (2.12) this is "precisely the desired five-dimensional Einstein equation, with the right-hand side being the energy-momentum tensor of the Yang-Mills fields" (p. 9). Note the sign of the F F term differs from (2.6) because the 5-form contribution (2.9) has been added.

**(c) Consistency condition.** Eq. (2.12), p. 9: `Y^{IJ} = β δ^{IJ}` with β a constant, i.e. `Y^{IJ}` in (2.11) must be independent of the internal coordinates y (then a change of basis of Killing vectors brings it to β δ^{IJ}). Interpretation (p. 9): any y-dependence of the right side of (2.10) means bilinears in F^I source massive spin-2 modes, so setting those to zero is inconsistent. Argument that the criterion is exact, not just linearised (p. 10/PDF 11, eqs. 2.13–2.14): the coupling `L_int = H^{IJμν}(F^I_{μρ}F^{Jρ}_ν − ¼ g_{μν} F^I_{σρ}F^{Jσρ})` to a massive spin-2 field `H^{IJ}_{μν}` must involve a conserved current; the bilinear (2.14) is the unique such current, so higher-order terms in the ansatz cannot cancel it. Rewritten in section 5 (eq. 5.1, p. 25/PDF 26, with `Λ̂ = D m²`, internal dimension D+1) and, using `□K̂^m + Λ̂ K̂^m = 0` and the identity (5.2)–(5.3), in the form (5.4), p. 26 (PDF 27):

  `Y(K̂, L̂) = ½ (D+2) K̂^m L̂_m + (D/(4Λ̂)) □(K̂^m L̂_m)`.

"The Consistency Condition" = Y(K̂, L̂) constant for every pair of retained Killing vectors. For D = 11 supergravity on seven-dimensional Einstein spaces the identical form holds with `R_{mn} = 6m² g_{mn}` (p. 10, citing [28] Duff–Nilsson–Pope–Warner 1984).

**(d) Spheres S^n and S^2.** Statement (p. 10/PDF 11): "for any round sphere S^n with `R_{mn} = (n−1) m² g_{mn}`, the condition (2.12) is satisfied by all the SO(n+1) Killing vectors. (This does not necessarily mean that a consistent Kaluza-Klein reduction on S^n is possible, though.)" Two proofs in section 5 (p. 26–27/PDF 27–28): (i) via Killing spinors, `∇̂_m η^A − (i/2) m Γ_m η^A = 0` (5.5), `K̂^{AB}_m = η̄^A Γ_m η^B`, `∇̂_m K̂^{AB}_n = i m η̄^A Γ_{mn} η^B`, plus Fierz rearrangements (details not shown); (ii) via the embedding in R^{n+1}: with (5.6)–(5.8),

  `K_{AB} = x^A ∂_B − x^B ∂_A`,  `ds²(R^{n+1}) = dr² + r² du^A du^A`,  `g_{AB} = (1/r²)(δ_{AB} − x^A x^B/r²)`,

  `(K_{AB} · K_{CD}) = δ_{AC} u_B u_D + δ_{BD} u_A u_C − δ_{AD} u_B u_C − δ_{BC} u_A u_D`  (5.9, p. 27),
  `□_{R^{n+1}}(r²(K_{AB}·K_{CD})) = 4(δ_{AC}δ_{BD} − δ_{AD}δ_{BC})`  (5.11),
  `□_{S^n}(K_{AB}·K_{CD}) + 2(n+1)(K_{AB}·K_{CD}) = 4(δ_{AC}δ_{BD} − δ_{AD}δ_{BC})`  (5.12),
  `Y(K_{AB}, K_{BC}) = δ_{AC}δ_{BD} − δ_{AD}δ_{BC}`  (5.13, on the unit S^n, m² = 1; the left side is a typo for Y(K_{AB}, K_{CD})).

For S^2 (n = 2, D = 1, Λ̂ = 1): the SO(3) Killing vectors satisfy the condition, `Y^{IJ} ∝ δ^{IJ}`. So, within the scope of this paper (Freund–Rubin-type backgrounds, gauge bosons from Killing vectors), nothing in the spin-2 sector obstructs the SU(2) ≅ SO(3) truncation on S^2. The paper does NOT analyse pure gravity on M_4 × S^2 and does not claim the S^2 truncation is consistent in general: footnote 2 (p. 3/PDF 4) records that a group-theoretic argument in [7, 10] shows an n-sphere reduction of gravity + dilaton + n-form keeping all SO(n+1) Yang-Mills fields is inconsistent except for listed exceptional cases, and p. 2–3 lists the consistent S^2 case as "certain theories of gravity plus a dilaton and a 2-form field strength in D dimensions on a 2-sphere [10]" (Cvetič–Lü–Pope hep-th/0003286). S^2 reappears as CP^1 in section 5 (p. 31–32/PDF 32–33): the three SO(3) eigenfunctions ψ on CP^1 satisfy `∇_a ∇_b ψ = −Λ g_{ab} ψ` and `∫|∇ψ|⁴ = (8/3) Λ² ∫ψ⁴` (5.35), so `∫|∇Q|² = (8/3) Λ (β+Λ)² ∫ψ⁴` (5.36) and Q is constant iff β = −Λ, which happens only when the U(1) fibre winds solely over that S^2 (total space S^3 × rest).

**(e) Explicit Killing vectors.** On S^n in Cartesian coordinates of R^{n+1}: `K_{AB} = x^A ∂_B − x^B ∂_A` (5.6, p. 26/PDF 27), with inner product (5.9). No polar-coordinate (θ, φ) expressions for S^2 are given. On CP^n in inhomogeneous coordinates ζ^α (4.18)–(4.20), p. 22 (PDF 23): `ψ = (1/r²) T_A{}^B Z^A Z̄_B = f^{-1}(T_0{}^0 + T_0{}^α ζ̄^{ᾱ} + T_α{}^0 ζ^α + T_α{}^β ζ^α ζ̄^{β̄})`, `f = 1 + ζ^α ζ̄^{ᾱ}`, `K^α = i g^{αβ̄} ∂_{β̄} ψ = (i/2)(T_0{}^α + T_β{}^α ζ^β − T_0{}^0 ζ^α − T_β{}^0 ζ^β ζ^α)`, T Hermitean traceless; cross-checked against `δζ^α = i ε(...)` from SU(n+1) acting on Z^A (4.21). Setting n = 1 gives the SO(3) Killing vectors of S^2 = CP^1 in a stereographic complex coordinate (metric `dΣ_1² = f^{-2} dζ dζ̄`, Λ = 4, i.e. radius ½).

**(f) Kähler-space results.** Lifting a Killing vector K of the base to the U(1) bundle `dŝ² = c²(dz − A)² + ds²` (3.1): `K̂ = K + h ∂_z` (3.4) with `∂_a h = L_K A_a`, `∂_z h = 0` (3.6–3.7), always solvable if F = dA is K-invariant (3.9–3.10). On a Kähler space with b_1 = 0 every Killing vector is `K^a = J^{ab} ∂_b ψ` (3.17); if Einstein, ψ is a Laplacian eigenfunction `□ψ + 2Λψ = 0` (3.19), and conversely (3.20); `ψ = (1/(2Λ)) J^{ab} ∇_a K_b` (3.26). With `F = αJ` the lift is `K̂ = K + (αψ + K^b A_b) ∂_z` (3.24), as a 1-form `K̂ = −i(∂ − ∂̄)ψ + α c² ψ (dz − A)` (3.25). Bundle Ricci: `R̂_{ab} = (Λ − ½c²α²) δ_{ab}`, `R̂_{zz} = ¼ c² α² D`, `R̂_{az} = 0` (3.21), Einstein iff `Λ = ¼ c² α² (D+2)` (3.22). Product bases: (3.28)–(3.46); Einstein metrics always exist for any non-zero winding numbers (3.41)–(3.43); the "supersymmetric" case `q_i = k_i/ℓ` has `Λ = ((D+2)/D) Λ̂`, `c²α² = 4Λ/(D+2)` (3.44–3.45) and two Killing spinors (4.33). Consistency sectors (5.17)–(5.23): `Y(U, K̂_i)` constant requires `(D+2)Λ̂ − DΛ_i = 0` (5.18); `Y(K̂_i, K̂_j)` for different factors requires `C_{ij} = (D+2)Λ̂ − D(Λ_i + Λ_j) = 0` (5.22), which equals −DΛ in the supersymmetric case (5.23) — already fatal; same-factor sector reduces to constancy of `Q = ∂^a ψ ∂_a ψ − β ψ²` (5.31) with β in (5.29)–(5.30), and `∫|∇Q|² = (8/3)Λ(β−Λ)² ∫ψ⁴ + 2(2β − ((d−2)/(d+2))Λ) ∫|∇ψ|⁴` (5.33), shown positive except in the trivial S^{4q+3} × (Kähler) cases (5.34)–(5.46), using the bounds (B.17)–(B.18).

# Attributable statements

1. The necessary condition for consistently retaining the Yang-Mills gauge bosons of the isometry group in a massless truncation on an Einstein internal space is that `Y^{IJ} = K^I_m K^{Jm} + (1/2m²) ∇_m K^I_n ∇^m K^{Jn}` be independent of the internal coordinates (section 2, eqs. 2.10–2.12, p. 9/PDF 10).
2. A y-dependent Y^{IJ} means bilinears in the gauge field strengths act as sources for massive spin-2 fields, so setting the massive spin-2 fields to zero is inconsistent (section 2, p. 9/PDF 10).
3. The condition derived from the linearised gauge-boson ansatz is exact: the bilinear current (2.14) coupling to massive spin-2 fields is the unique conserved one, so no higher-order terms in the ansatz can cancel the inconsistency (section 2, p. 10/PDF 11; citing Pope–Stelle [26]).
4. All SO(n+1) Killing vectors of any round sphere S^n satisfy the consistency condition, with `Y(K_{AB}, K_{CD}) = δ_{AC}δ_{BD} − δ_{AD}δ_{BC}` on the unit sphere; this does not by itself imply that a consistent reduction on S^n exists (p. 10/PDF 11 and section 5, eqs. 5.6–5.13, p. 26–27/PDF 27–28).
5. For the metric ansatz (2.3), the higher-dimensional Ricci tensor and curvature scalar are given exactly by (2.6)–(2.8): `R̂ = R_{(5)} + R_{(M)} − ¼ K^I_a K^{Ja} F^I_{αβ} F^{Jαβ}` (section 2, p. 8/PDF 9).
6. Inconsistencies, when present, show up at the level of cubic interactions bilinear in the lower-dimensional gauge fields coupling to linearised spin-2 fields; this sector gives a necessary condition for consistency (Introduction, p. 4–5/PDF 5–6).
7. For the U(1) bundles Q^{q_1...q_N}_{n_1...n_N} over products of CP^{n_i} (including T^{1,1} = Q^{11}_{11}, M^{3,2} = Q^{32}_{21}, Q^{111}_{111}) the non-abelian SU(n_i+1) Killing vectors never satisfy the consistency condition; only the U(1) gauge field of the fibre can be kept (abstract; section 5, p. 34/PDF 35; Conclusions p. 35/PDF 36).
8. Spheres work in supergravity reductions because they maximise the number of Killing spinors, so that the supergravity multiplet (as opposed to matter multiplets) contains all the Yang-Mills fields of the isometry group (Introduction, p. 6/PDF 7, following Pope [27]).
9. On an Einstein–Kähler space with b_1 = 0 there is a one-to-one correspondence between Killing vectors `K^a = J^{ab} ∂_b ψ` and scalar Laplacian eigenfunctions with eigenvalue 2Λ (section 3.2, eqs. 3.17–3.20, p. 14–15/PDF 15–16).
10. Every Killing vector of the base of a U(1) bundle with G-invariant curvature lifts to a Killing vector of the bundle, so the bundle isometry group contains G × U(1) (section 3.1, p. 11–13/PDF 12–14).

# Caveats

- Scope mismatch with the tutorial: the paper's setting is type IIB on AdS_5 × M_5 (and D = 11 on AdS_4 × M_7) with Freund–Rubin flux, not pure gravity on M_4 × S^2. Pure Einstein gravity has no product vacuum with a round S^2 factor without flux or cosmological constant, so the paper's consistency criterion cannot be transplanted literally; only the geometric content (ansatz (2.3), Ricci tensor (2.6)–(2.8), Killing-vector identity (5.13) on S^n) carries over directly.
- The ansatz (2.3) contains only gauge bosons: no warp factor, no breathing mode or other scalars (footnote 3, p. 7/PDF 8, explicitly says truncating the scalars while keeping the gauge fields is itself inconsistent, but a separate inconsistency). A tutorial ansatz with a scale factor φ(x) multiplying the internal metric is more general than (2.3).
- Sign convention: the gauge field enters as `e^a − K^{Ia} A^I`, i.e. `dy^m − K^{Im} A^I`. Many texts write `dy^m + K^m_i A^i`; the relative sign correlates with the sign in `F = dA + ½ f A∧A` and with `[K^I, K^J] = +f^{IJ}_K K^K`. No gauge coupling g is written.
- "Satisfies the consistency condition" (spheres) is explicitly only necessary, not sufficient (p. 10/PDF 11). The paper's positive claims about actual consistent S^2 reductions are citations to [10] (gravity + dilaton + 2-form), not results proved here.
- Apparent typos: p. 27 (PDF 28) says "the unit S^n has cosmological constant (n+1)", but with their convention `R_{mn} = (n−1) m² g_{mn}` (p. 10) and `Λ̂ = D m²`, D + 1 = n, the unit sphere has Λ̂ = n − 1 and m² = 1 (which is what (5.12)–(5.13) actually use: substituting D = n − 1, Λ̂ = n − 1 into (5.4) reproduces (5.13)). Also (5.13) is written `Y(K_{AB}, K_{BC})` where `Y(K_{AB}, K_{CD})` is meant. (3.29) has a stray double comma. In (3.21) the text says "if a is chosen" where α is meant.
- The Riemann/Ricci sign convention is inferred (see Conventions), not declared. The Killing-spinor proof for S^n (5.5) is sketched only ("Fierz rearrangements"); the R^{n+1} embedding proof (5.6)–(5.13) is complete.
- Equation (5.26)–(5.27) numbering: (5.27) appears in footnote 8 (p. 30/PDF 31), between (5.26) and (5.28) in the main text.
- Y^{IJ} = β δ^{IJ} fixes the relative normalisation of the Killing vectors but not their overall scale (β is arbitrary); the physical gauge coupling in the reduced theory is therefore tied to β and to the radius of the internal space, which the paper does not spell out.
- "Linearised" is used in two senses: the ansatz (2.3), (2.9) is linear in the gauge fields, but the curvatures (2.6)–(2.8) and the Einstein equation (2.10) are computed exactly for that ansatz and are quadratic in F. Do not read (2.8) as an approximation.
- The paper derives only the spacetime components of the higher-dimensional Einstein equation. It does not write a reduced Yang-Mills equation, a reduced action, or the (ab) and (αb) components as lower-dimensional equations; the internal curvature scalar `R_{(M)}` in (2.10) is a constant (Einstein space) and plays the role of a cosmological term.
- The Killing vectors in (2.3) are those of the full isometry group G of M_5, with no further assumption such as homogeneity or a coset structure; the sphere and Q-space results are special cases, and the general machinery (lifting to U(1) bundles) is tailored to Kähler bases, so none of the coset-space (G/H) language of the tutorial appears here.
- Reference numbering in the PDF is by number; the key companions are [10] Cvetič–Lü–Pope hep-th/0003286 (consistent sphere reductions), [26] Pope–Stelle 1987 (zilch currents, used for the exactness argument), [28] Duff–Nilsson–Pope–Warner 1984 (the D = 11 analogue of (2.11)), and [1] Duff–Nilsson–Pope Phys. Rep. 130 (1986) (the linearised harmonic analysis).
