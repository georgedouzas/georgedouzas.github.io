# Identity

- arXiv: hep-th/0409191 (v2, 21 Dec 2004). Report numbers HUTP-03/A053, RUNHETC-2003-26.
- Title: *M theory and Singularities of Exceptional Holonomy Manifolds*
- Authors: Bobby S. Acharya (ICTP Trieste), Sergei Gukov (Harvard)
- Year: 2004 (dated September 2004). Venue: not stated in the PDF (a review article; my recollection is Phys. Rept. 392 (2004) 121, but this does not appear in the file and should be checked before citing).
- Length: 94 printed pages; PDF has 95 pages. **Printed page = PDF page − 1** throughout.

# Structure

PDF page in brackets, printed page after it.

1. Introduction — PDF 3–6 / printed 2–5
2. Riemannian Manifolds of Special Holonomy — PDF 7 / p.6
   - 2.1 Holonomy Groups (p.6); 2.2 Relation Between Holonomy and Supersymmetry (p.8); 2.3 Invariant Forms and Minimal Submanifolds (p.10); 2.4 Why Exceptional Holonomy is Hard (p.14)
3. Construction of Manifolds With Exceptional Holonomy — PDF 17 / p.16
   - 3.1 Compact Manifolds (p.16, Joyce orbifolds, Kovalev); 3.2 Non-compact Manifolds (p.20, cones, Hitchin flow, Bryant–Salamon)
4. M theory on Smooth Special Holonomy Manifolds — PDF 29–33 / p.28–32 (the Kaluza–Klein reduction; the core section for the tutorial)
5. M theory Dynamics on Singular Special Holonomy Manifolds — PDF 34 / p.33
   - 5.1 Low Energy Dynamics via Duality with the Heterotic String (p.34; ADE singularities, pp.36–44); 5.2 Low Energy Dynamics via IIA Duals (p.44; D6-branes, Taub–NUT, conical G_2 example)
6. Chiral Fermions from Conical Singularities in G_2-manifolds — PDF 52 / p.51
   - 6.1 Hints from Anomaly-Inflow (p.51); 6.2 Chiral Fermions via Duality With The Heterotic String (p.53); 6.3 Other Gauge Groups and Matter Representations (p.60) [not read in detail]
7. Topology Change in M theory on Exceptional Holonomy Manifolds — p.63; 7.1 (p.63); 7.2 Relation to Geometric Transition (p.69) [skipped]
8. Quantum Super-Yang-Mills from G_2 Manifolds — PDF 76 / p.75
   - 8.1 SYM review (p.75); 8.2 Theta angle and Coupling Constant in M theory (p.77); 8.3 Superpotential (p.77); 8.4 ADE-singular G_2 (p.84); 8.5 Confinement (p.85); 8.6 Mass Gap (p.86)
- Acknowledgments p.87; References [1]–[110] pp.88–94.

# Conventions

- Signature: never stated explicitly. 4D spacetime is written as Minkowski space $\mathbb{R}^{3,1}$ (also $\mathbb{R}^{6,1}$, $\mathbb{R}^{2,1}$); the internal $X$ is Riemannian, $ds^2=\sum_{i=1}^7 e^i\otimes e^i$ (2.15). Backgrounds are $\mathbb{R}^{11-n}\times X$ (p.10).
- 11D supergravity action: **no normalised action is written.** Section 4 (p.28) states only: bosonic fields are the metric $g$ and 3-form $C$ with $G=dC$, and "the equation of motion for $C$ is $d*G=\tfrac12 G\wedge G$". The Chern–Simons term is quoted as $S=\int_{X\times\mathbb{R}^{3,1}}\tfrac16 C\wedge G\wedge G$ (6.100, p.52) with footnote 11: "We will not be too careful about factors in this section." Membrane flux quantisation: $[G/2\pi]-\lambda/2\in H^4(X;\mathbb{Z})$, $\lambda=p_1/2$ (4.62, p.31).
- $G_2$ 3-form: $\Phi=\tfrac{1}{3!}\psi_{ijk}\,dx_i\wedge dx_j\wedge dx_k$ (2.6) with $\psi_{ijk}$ the imaginary-octonion structure constants $\sigma_i\sigma_j=-\delta_{ij}+\psi_{ijk}\sigma_k$ (2.7), and in a particular basis $\psi_{ijk}=+1$ for $(ijk)\in\{(123),(147),(165),(246),(257),(354),(367)\}$ (2.8). $G_2\subset GL(7,\mathbb{R})$ is defined as the stabiliser of $\Phi$. The coassociative 4-form is $*\Phi$. (The paper writes $\Phi$, not $\varphi$.) The $T^7$ example on p.16 uses a *different* sign pattern: $\Phi=e^{123}+e^{145}+e^{167}+e^{246}-e^{257}-e^{347}-e^{356}$.
- Spin(7) Cayley 4-form (2.9): $\Omega=e^{1234}+e^{1256}+e^{1278}+e^{1357}-e^{1368}-e^{1458}-e^{1467}-e^{2358}-e^{2367}-e^{2457}+e^{2468}+e^{3456}+e^{3478}+e^{5678}$.

# Equations the tutorial uses

**(a) $G_2$ holonomy via the associative 3-form** (Sec. 2.3, p.12):
- (2.14): $d\Phi=0,\quad d*\Phi=0$ — "the existence of $G_2$ holonomy metric on $X$ is equivalent to the closure and co-closure of the associative form". Footnote 1: equivalently the $G_2$-structure $(g,\Phi)$ is torsion-free, $\nabla\Phi=0$.
- (2.16): $\Phi=\tfrac{1}{3!}\psi_{ijk}\,e^i\wedge e^j\wedge e^k$ in terms of the vielbein of (2.15).
- (2.17): the metric is reconstructed from $\Phi$: $g_{ij}=\det(B)^{-1/9}B_{ij}$, $\;B_{jk}=-\tfrac{1}{144}\Phi_{ji_1i_2}\Phi_{ki_3i_4}\Phi_{i_5i_6i_7}\epsilon^{i_1\ldots i_7}$.
- (2.12): invariant forms from the parallel spinor, $\omega^{(p)}=\xi^\dagger\Gamma_{i_1\ldots i_p}\xi$; on a $G_2$ manifold these exist only for $p=3,4$ (associative and coassociative).
- (2.13): $G_2$ decomposition of cohomology: $H^2=H^2_{\mathbf 7}\oplus H^2_{\mathbf{14}}$, $H^3=H^3_{\mathbf 1}\oplus H^3_{\mathbf 7}\oplus H^3_{\mathbf{27}}$, $H^4=H^4_{\mathbf 1}\oplus H^4_{\mathbf 7}\oplus H^4_{\mathbf{27}}$, etc.; for compact irreducible $G_2$ holonomy $H^k_{\mathbf 7}(X,\mathbb{R})=0,\ k=1,\ldots,6$ (unnumbered, p.11), in particular $b_1=0$.
- Berger's list is Table 1 (p.7): $U(n/2)$, $SU(n/2)$, $Sp(n/4)$, $Sp(n/4)Sp(1)$, $G_2$ ($n=7$), $Spin(7)$ ($n=8$), $Spin(9)$ ($n=16$).

**(b) Supersymmetry argument** (Sec. 2.2, pp.9–10):
- (2.10): $\nabla\xi=0$. "With all bosonic fields apart from the metric set to zero, 2.10 is precisely the condition for unbroken supersymmetry in string or M theory compactification. This condition on a spinor field automatically implies a holonomy reduction: since $\xi$ is invariant under parallel transport, $Hol(g(X))$ must be such that the spinor representation contains the trivial representation. This is impossible if $Hol(g(X))$ is $SO(n)$ ... Therefore $Hol(g(X))\subset SO(n)$." For $Hol=G_2$: $\mathbf 8\to\mathbf 7\oplus\mathbf 1$ (unnumbered, p.10). All such manifolds are Ricci-flat, $R_{ij}=0$ (unnumbered, p.10), so $\mathbb{R}^{11-n}\times X$ solves the 11D Einstein equations with no sources. Table 2 (p.9): $T^n$ (SUSY 1) $\supset$ CY$_3$ ($1/4$) $\supset$ $G_2$ ($1/8$) $\supset$ $Spin(7)$ ($1/16$). (2.11): M on $G_2\Rightarrow D=3+1$ QFT; M on $Spin(7)\Rightarrow D=2+1$ QFT, both minimal SUSY.

**(c) Massless 4D spectrum from M theory on smooth $X$** (Sec. 4, pp.28–31; attributes the analysis to Papadopoulos–Townsend [44]):
- $C$-field scalars, (4.46): $C=\sum\omega^I(x)\phi_I(y)+\ldots$, $\omega^I$ a basis of the $b_3(X)$ harmonic 3-forms; (4.47) $G=\sum\omega^I\wedge d\phi_I$; (4.48) $d*G=\sum *_7\omega^I\wedge d*_4 d\phi_I$; since $G\wedge G$ vanishes to first order, the $\phi_I$ are massless: "$b_3(X)$ real massless scalars".
- Metric moduli: (4.49) $R_{ij}(g(X))=0$; (4.50) $g_{ij}(x)+\delta g_{ij}(x,y)$; Lichnerowicz (4.51) $\Delta_L\delta g_{ij}\equiv-\nabla^2_M\delta g_{ij}-2R_{imjn}\delta g^{mn}+2R^k_{(i}\delta g_{j)k}=0$; KK ansatz (4.52) $\delta g_{ij}=h_{ij}(x)\rho(y)$; (4.53) $\nabla^2_M=\nabla^2_\mu+\nabla^2_i$; (4.54) $h_{ij}\nabla^2_\mu\rho=-(\Delta_L h_{ij})\rho=-\lambda h_{ij}\rho$, so 4D mass$^2$ = Lichnerowicz eigenvalues. Counting: $h_{ij}\in\mathbf{27}$ of $SO(7)$, irreducible under $G_2$; 3-forms $\mathbf{35}\to\mathbf 1+\mathbf 7+\mathbf{27}$ (4.55); map (4.56) $\Phi_{n[pq}h^n_{r]}=\omega_{pqr}$; (4.57) $\Delta_L h=0\leftrightarrow\Delta\omega=0$. Hence "precisely $b_3(X)$ additional massless scalar fields" from the metric.
- Complexified moduli: choose $\alpha_I$ a basis of $H_3(X,\mathbb{R})$ with (4.58) $\int_{\alpha_I}\omega^J=\delta^J_I$; fluctuating $G_2$ structure (4.59) $\Phi'=\Phi+\delta\Phi=\Phi+\sum\rho^I(y)\omega^I(x)$; then (4.60) $\Phi^I(y)=\int_{\alpha_I}(\Phi'+iC)$ — lowest components of $b_3(X)$ massless chiral superfields.
- Vectors, (4.61): $C=\sum\beta^\alpha(x)\wedge A_\alpha(y)$, $\beta^\alpha$ a basis of harmonic 2-forms ⇒ $b_2(X)$ massless abelian gauge fields; fermionic partners from the gravitino. No 4D 2-forms since $b_1(X)=0$.
- Summary (p.31): "The low energy effective theory is an $\mathcal N=1$ supergravity theory coupled to $b_2(X)$ abelian vector multiplets and $b_3(X)$ massless, *neutral* chiral multiplets."
- Non-compact $X$ (p.32): replace $b_k$ by the dimension of $L^2$-normalisable harmonic $k$-forms.

**(d) Why singularities are needed, and what gives what:**
- p.4 (Intro): "if $X$ is smooth, the four dimensional physics contains at most an abelian gauge group and no light charged particles." p.31: "the gauge group is abelian and there are no light charged particles... these features [non-Abelian gauge fields and chiral fermions] emerge naturally from singularities in $G_2$-manifolds."
- p.19: codimension-$d$ singularity ↔ physics of a $D\ge 11-d$ dimensional field theory; so 4D chiral matter cannot come from codimension-4 ($\mathbb{C}^2/\mathbb{Z}_2$) singularities, which give $D=7$ theories; one needs singularities of maximal codimension (isolated points) in 7 dimensions.
- p.27 (end of Sec. 3): isometries of non-compact $G_2$ cones are global symmetries only; "we can not use the geometric symmetries of $X$ to get non-abelian gauge fields" (contrast with coset KK).
- ADE ⇒ gauge group (Sec. 5.1, pp.36–44): $\mathbb{C}^2/\Gamma_{ADE}$, $\Gamma\subset SU(2)$ finite (generators (5.69)–(5.73)); M theory on $\mathbb{C}^2/\Gamma_{ADE}\times\mathbb{R}^{6,1}$ is 7D SYM with ADE gauge group (p.41; W-bosons = M2-branes on the vanishing 2-spheres). In a $G_2$ manifold, ADE singularities along a 3-manifold $W$ ($X\simeq\mathbb{C}^2/\Gamma_{ADE}\times W$ locally, with $\Phi=\tfrac16\omega_i\wedge e_j\delta^{ij}+e_1\wedge e_2\wedge e_3$ (5.89)) give 4D $\mathcal N=1$ SYM with $b_1(W)$ adjoint chiral multiplets (p.43–44); $b_1(W)=0$ (e.g. $W=S^3$) gives pure SYM.
- Conical ⇒ chiral multiplets (Sec. 6, pp.51–56): cone metric (3.26) $ds^2(X)=dt^2+t^2ds^2(Y)$; anomaly inflow (6.99)–(6.105) from $\int\tfrac16C\wedge G\wedge G$ gives $\delta S\sim\int_Y\beta^\rho\wedge\beta^\sigma\wedge\beta^\delta\int_{\mathbb{R}^{3,1}}\epsilon^\rho dA^\sigma\wedge dA^\delta$, cancelled only by chiral fermions at the conical point; for ADE, chiral fermions sit at points $P\in W$ where the singularity "worsens" (rank increases), (6.110) $dK=2\pi q\,\delta_P$, (6.111) $\delta S\sim-q\int tr\lambda F\wedge F$. Sec. 6.2: SU(5) example from $E_8\times E_8$ heterotic, unfolding (6.112) $xy=(z+5\epsilon)(z-\epsilon)^5$.
- 4D gauge coupling (Sec. 8.2, p.77): (8.137) $1/g^2_{7d}\sim1/l_p^3$; (8.139) $1/g^2_{4d}\sim V_W/l_p^3$; (8.140) $\theta=\int_W C$; (8.141) $\tau=\int_W\frac{C}{2\pi}+i\frac{\Phi}{l_p^3}$.

**(e) Relation to Calabi–Yau / IIA / circle fibrations:**
- No statement of the form "$G_2=CY_3\times S^1$" appears. The explicit fibration statements are:
  - Kovalev (p.20): $X\cong(W_1\times S^1)\cup(W_2\times S^1)$ with $W_{1,2}$ asymptotically cylindrical CY 3-folds glued back-to-back (smooth compact $G_2$).
  - Sec. 5.2 (p.44): IIA $\Leftrightarrow$ M on $S^1$ with (5.90) $ds^2=e^{-\frac23\phi}g_{ij}dx^idx^j+e^{\frac43\phi}(dx_{11}+A_idx^i)^2$; Taub–NUT (5.91) $ds^2_{TN}=H\,d\vec x^{\,2}+H^{-1}(dx_{11}+A_idx^i)^2$, $\nabla\times\vec A=-\nabla H$, $H=1+\frac{1}{2|\vec x|}$. General: "M theory on $X$ $\Leftrightarrow$ IIA on $X/U(1)$ with D6-branes on $L$" ($L$ the codimension-4 fixed set, p.46); for non-compact $G_2$ with (5.92) $X/U(1)\cong\mathbb{R}^6$, $L$ is special Lagrangian and (5.93) $b_k(L)=b_{k+2}(X)$ ($k>0$), $b_0(L)=b_2(X)+1$.
  - Sec. 6.2 (p.54): heterotic on a $T^3$-fibred CY $Z\to W$ is dual to M theory on a K3-fibred 7-manifold $X\to W$, which is a $G_2$ manifold by supersymmetry (this is the "$K3$-fibred $G_2$ ↔ $T^3$-fibred CY" duality of p.4).
  - Sec. 8.3 (p.78): IIA on a smooth $G_2$ $X$ gives 3D $\mathcal N=2$ sugra with $b_2(X)+b_3(X)$ scalar multiplets ($b_3$ from $g$ and $A_3$; $b_2$ from $B$ and $A_3$).

**(f) Explicit $G_2$ examples:**
- Joyce orbifold $T^7/(\mathbb{Z}_2)^3$ (pp.16–18): $\Phi$ as in Conventions, involutions $\alpha:(x_1,\ldots,x_7)\mapsto(x_1,x_2,x_3,-x_4,-x_5,-x_6,-x_7)$, $\beta:\mapsto(x_1,-x_2,-x_3,x_4,x_5,\tfrac12-x_6,-x_7)$, $\gamma:\mapsto(-x_1,x_2,-x_3,x_4,\tfrac12-x_5,x_6,\tfrac12-x_7)$ (3.22); resolved manifold has (3.24) $b^2(X)=12$, $b^3(X)=43$, computed as $b^2=b^2_\Gamma(T^7)+12\cdot1$, $b^3=b^3_\Gamma(T^7)+12\cdot3$ with $b^3_\Gamma(T^7)=7$ (3.25).
- Asymptotically conical (Table 4, p.21): $S^4\times\mathbb{R}^3$ ($Y=\mathbb{CP}^3$), $\mathbb{CP}^2\times\mathbb{R}^3$ ($Y=SU(3)/U(1)^2$), $S^3\times\mathbb{R}^4$ ($Y=SU(2)\times SU(2)$), $T^{1,1}\times\mathbb{R}^2$. Cone bases are cosets $Y=G/K$ (3.28); $X\cong(G/H)\times\mathbb{R}^{k+1}$ when $H/K=S^k$ (3.31). Bryant–Salamon metric on the spin bundle of $S^3$, (3.45): $ds^2=\frac{dr^2}{1-r_0^2/r^2}+\frac{r^2}{12}\sum_{a=1}^3(\sigma_a-\Sigma_a)^2+\frac{r^2}{36}\left(1-\frac{r_0^2}{r^2}\right)\sum_{a=1}^3(\sigma_a+\Sigma_a)^2$, obtained from Hitchin's flow with $\Phi=dt\wedge\omega+\rho$ (3.44).

**Spin(7) paragraph.** Defined as the stabiliser in $GL(8,\mathbb{R})$ of the Cayley 4-form (2.9); $G_2$ holonomy metric ⇔ $d\Omega=0$ is implicit in "only one invariant form in degree 4" (p.12) but the Spin(7) torsion-free condition is not written as a numbered equation. Cohomology decomposition (2.18) and compact constraints (2.19) $H^1_{\mathbf 8}=H^3_{\mathbf 8}=H^5_{\mathbf 8}=H^7_{\mathbf 8}=0$, $H^2_{\mathbf 7}=H^4_{\mathbf 7}=H^6_{\mathbf 7}=0$. M theory on $Spin(7)$ gives 3D $\mathcal N=1$ (two supercharges) with $b_2$ abelian vectors and $(b_3+b_4^-+1)$ scalars (p.31); membrane anomaly may force half-integral $G$-flux (4.62), generating Chern–Simons levels (4.63) $k_{ij}=\frac{1}{2\pi}\int_X\beta_i\wedge\beta_j\wedge G$ and superpotential (4.64) $\mathcal W=\frac{1}{2\pi}\int_X\Omega\wedge G$; 3D action (4.65). Duality (5.98): non-compact $Spin(7)$ $X$ with $X/U(1)\cong\mathbb{R}^7$ ↔ IIA with D6 on coassociative $L\subset\mathbb{R}^7$. AC examples in Table 4: $S^4\times\mathbb{R}^4$, $\mathbb{CP}^2\times\mathbb{R}^4$, $S^5\times\mathbb{R}^3$. Calibrated submanifolds: Cayley 4-folds (Table 3).

# Attributable statements

1. "The 14-dimensional simple Lie group $G_2$ is precisely the automorphism group of the octonions" and may be defined as the subgroup of $GL(7,\mathbb{R})$ preserving the 3-form (2.6). — Sec. 2.1, p.7.
2. "manifolds with larger holonomy group preserve a smaller fraction of the supersymmetry" (Table 2); M theory on a $G_2$ manifold gives $\mathcal N=1$ in 4D, "similar to the compactification of heterotic string theory on Calabi-Yau three-folds". — Sec. 2.2, p.8–9.
3. $\nabla\xi=0$ (2.10) is the unbroken-SUSY condition with only the metric on, and forces $Hol\subsetneq SO(n)$; all the listed holonomies are Ricci-flat so $\mathbb{R}^{11-n}\times X$ solves the 11D Einstein equations. — Sec. 2.2, p.9–10.
4. $G_2$ holonomy ⇔ $d\Phi=d*\Phi=0$ (2.14), equivalently $\nabla\Phi=0$; the metric is determined by $\Phi$ via (2.17). — Sec. 2.3, p.12.
5. No analogue of Yau's theorem exists for $G_2$/$Spin(7)$; "to date, not a single example of a Ricci flat metric of special holonomy is known *explicitly* for a compact, simply connected manifold!" — Sec. 2.4, p.15.
6. KK reduction of 11D sugra on smooth $G_2$ $X$: $b_3(X)$ real scalars from $C$ (4.46), $b_3(X)$ from Lichnerowicz zero modes of $g$ (4.57), combining into complex scalars $\Phi^I=\int_{\alpha_I}(\Phi'+iC)$ (4.60); $b_2(X)$ abelian vectors from (4.61); result: $\mathcal N=1$ sugra + $b_2$ vector multiplets + $b_3$ neutral chiral multiplets. — Sec. 4, pp.28–31, citing Papadopoulos–Townsend [44].
7. "if $X$ is smooth, the four dimensional physics contains at most an abelian gauge group and no light charged particles"; non-abelian gauge symmetry and chiral fermions require singularities. — Intro p.4; Sec. 4 p.31.
8. A codimension-$d$ singularity is associated with a $D\ge 11-d$ field theory; 4D chiral matter needs isolated (codimension-7) singular points, ADE codimension-4 loci give 7D SYM. — Sec. 3.1, p.19.
9. M theory on $\mathbb{C}^2/\Gamma_{ADE}\times\mathbb{R}^{6,1}$ is 7D SYM with ADE gauge group (from M/heterotic duality on K3/$T^3$); ADE singularities along a supersymmetric 3-cycle $W$ in a $G_2$ manifold give 4D $\mathcal N=1$ SYM with $b_1(W)$ adjoint chirals. — Sec. 5.1, pp.41–44.
10. Chiral fermions arise at conical singularities, e.g. where an $A_n$ singularity along $W$ enhances rank at a point $P$; anomaly inflow (6.99)–(6.111) and heterotic duality (6.2) establish this. — Sec. 6, pp.51–56.
11. Continuous isometries of non-compact $G_2$ cones are global symmetries only; one "can not use the geometric symmetries of $X$ to get non-abelian gauge fields". — Sec. 3.2, p.27.
12. 4D gauge coupling and theta angle from ADE locus $W$: $1/g^2_{4d}\sim V_W/l_p^3$, $\theta=\int_W C$, $\tau=\int_W C/2\pi+i\Phi/l_p^3$. — Sec. 8.2, p.77.

# Caveats

- Printed page numbers are PDF page − 1; equation numbers run continuously (first numbered equation in Sec. 2 is (2.5)).
- The 11D action is never written with a normalisation; only $d*G=\tfrac12G\wedge G$ (p.28) and the CS term $\int\tfrac16C\wedge G\wedge G$ (6.100) appear, and the authors explicitly disclaim care about factors in Sec. 6. Do not cite this paper for the coefficient of the CS term.
- Signature is not stated; the reduction in Sec. 4 is at the level of equations of motion and KK ansätze, not an action, so no $\sqrt{-g}$ conventions appear.
- Two different $\psi_{ijk}$ sign/index conventions are used: (2.8) and the $T^7$ example (p.16). The latter matches Joyce's convention; the former matches the octonion table chosen in (2.7). Both define $G_2$ structures but are not identical 3-forms.
- The complexification (4.60) is written $\int(\Phi'+iC)$; much of the later literature uses $C+i\Phi$. The normalisation exponent in (2.17) is $-1/9$; other references give the same formula with different overall constants.
- In (4.55) the $\mathbf{35}$ of $SO(7)$ is the 3-form representation; the $\mathbf{27}$ of $G_2$ in $h_{ij}$ is the traceless symmetric tensor — the trace (volume) mode is the $\mathbf 1$ in $H^3_{\mathbf 1}$ (the class of $\Phi$ itself). The paper does not separate these explicitly.
- The count "$b_3(X)$" chiral multiplets uses $H^3_{\mathbf 7}=0$ (compact, full $G_2$ holonomy); for non-compact $X$ use $L^2$ forms (p.32).
- Stale cross-references: p.35 "a la equation (33)" should be (4.61); p.21 "Table 3" for the AC metrics should be Table 4.
- Spin(7) 3D spectrum $(b_3+b_4^-+1)$ scalars is quoted without derivation; the $+1$ is the overall volume (Cayley-form class $H^4_{\mathbf 1^+}=\langle[\Omega]\rangle$).
- The $G_2\Rightarrow$ SUSY argument (2.10) is stated for pure-metric backgrounds; with $G$-flux (Spin(7) case, Sec. 4) the condition is modified and a superpotential appears (4.64).
- The paper gives no $G_2$ example of the form $CY_3\times S^1$ (holonomy $SU(3)\subsetneq G_2$); the Kovalev gluing (p.20) and the K3-fibration (p.54) are the only fibration statements.
