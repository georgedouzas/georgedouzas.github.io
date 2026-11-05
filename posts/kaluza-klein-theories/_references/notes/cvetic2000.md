# Identity

- arXiv: hep-th/0003286 (v2, 4 May 2000); preprint numbers CTP TAMU-10/00, UPR/881-T.
- Title: Consistent Kaluza-Klein Sphere Reductions.
- Authors: M. Cvetič, H. Lü (Univ. of Pennsylvania), C. N. Pope (Texas A&M).
- Year: 2000 (March 2000). No journal venue stated in the PDF (published later in Phys. Rev. D 62 (2000) 064028; not stated in the file).
- PDF page = printed page + 1 throughout (printed page 1 is PDF page 2).

# Structure

1. Introduction — PDF 2-6 (printed 1-5). Survey of known consistent sphere reductions; group-theoretic criterion; plan.
2. Possibilities for SO(n+1) Kaluza-Klein reductions on S^n — PDF 6-11 (printed 5-10). Dilaton-vector / δ-parameter argument, eqs. (1)-(12), Table 1 on PDF 11 (printed 10).
3. Consistent S^3 reduction — PDF 11-14 (printed 10-13). Eqs. (13)-(29): Lagrangian, full ansatz, equations of motion, reduced (D-3)-dimensional Lagrangian.
4. Truncations to previous results — PDF 14-17 (printed 13-16). 4.1 SO(4) → SU(2) (eqs. (30)-(38)); 4.2 SO(4) → U(1)×U(1) (eqs. (39)-(45)).
5. S^{D-3} reduction and D=3, N=8 gauged supergravity — PDF 17-19 (printed 16-18). Eqs. (46)-(49).
6. S^2 reduction — PDF 19-20 (printed 18-19). Eqs. (50)-(60), including the lift to (D+1)-dimensional pure gravity on a topological S^3.
7. Conformal anomaly terms — PDF 20-23 (printed 19-22). Eqs. (61)-(71): cosmological term, which reductions survive.
8. Conclusions and discussions — PDF 23-26 (printed 22-25). Group-theory criterion as a proof of inconsistency; supersymmetry argument for fermions.
- Acknowledgement and References [1]-[33] — PDF 26-28 (printed 25-27).

# Conventions

- Signature: not stated explicitly. The Lagrangians are of the form $\hat R\,\hat{*}1 - \tfrac12 \hat{*}d\hat\phi\wedge d\hat\phi - \ldots$ (standard Lü-Pope differential-form conventions, mostly-plus signature implied; $\epsilon_{D-3}$, $\epsilon_3$, $\epsilon_{D-2}$ denote lower-dimensional volume forms). Hats denote D-dimensional quantities (remark after eq. (4), PDF 8).
- Higher-dimensional Lagrangian, eq. (3), PDF 8 (printed 7):
  $$\mathcal{L}_D = \hat R\,\hat{*}1 - \tfrac12\,\hat{*}d\hat\phi\wedge d\hat\phi - \tfrac12\,e^{-a\hat\phi}\,\hat{*}\hat F_{(p)}\wedge\hat F_{(p)},$$
  with the dilaton coupling fixed by eq. (4): $a^2 = 4 - \dfrac{2(p-1)(D-p-1)}{D-2}$ (i.e. $\delta=4$ in the parametrisation (1), $|\vec c|^2 = \delta - \frac{2(m-1)(D-m-1)}{D-2}$). For $p=2$: $a^2 = \frac{2(D-1)}{D-2}$ (eq. (6)); for $p=3$: $a^2=\frac{8}{D-2}$ (eq. (7)). The $p=3$ case, eq. (13), PDF 11, is the bosonic string low-energy Lagrangian.
- Sphere coordinates, eq. (15), PDF 12 (printed 11): embedding coordinates $\mu^i$ with
  $$\mu^i\mu^i = 1,\qquad \Delta = T_{ij}\mu^i\mu^j,\qquad U = 2T_{ik}T_{jk}\mu^i\mu^j - \Delta\,T_{ii},\qquad Y=\det(T_{ij}),$$
  indices $i,j,\ldots$ ranging over $n+1$ values (4 for $S^3$, $D-2$ for $S^{D-3}$, 3 for $S^2$); repeated SO(n+1) indices summed. $T_{ij}$ is a symmetric unimodular-up-to-$Y$ tensor; $\tilde T_{ij}$ is the unimodular part: $\tilde T_{ij}=Y^{1/4}T_{ij}$ for $S^3$ (eq. (22)), $T_{ij}=Y^{1/(D-2)}\tilde T_{ij}$ for $S^{D-3}$, $T_{ij}=Y^{1/3}\tilde T_{ij}$ for $S^2$.
- SO(n+1) Yang-Mills ansatz: gauge-covariant derivatives, eq. (16), PDF 12:
  $$\mathcal{D}\mu^i = d\mu^i + g\,A^{ij}_{(1)}\mu^j,\qquad \mathcal{D}T_{ij} = dT_{ij} + g A^{ik}_{(1)}T_{kj} + g A^{jk}_{(1)}T_{ik},$$
  with field strength eq. (17): $F^{ij}_{(2)} = dA^{ij}_{(1)} + g\,A^{ik}_{(1)}\wedge A^{kj}_{(1)}$. $A^{ij}_{(1)}$ are the SO(n+1) gauge potentials from the isometry group of the sphere; $g$ is the gauge coupling.
- Euler angles for $S^3$ (eq. (33), PDF 15): $\mu_0+i\mu_3=\cos\tfrac\theta2 e^{i(\psi+\phi)/2}$, $\mu_1+i\mu_2=\sin\tfrac\theta2 e^{i(\psi-\phi)/2}$; left-invariant 1-forms eq. (34): $\sigma_1+i\sigma_2 = e^{-i\psi}(d\theta+i\sin\theta\,d\phi)$, $\sigma_3=d\psi+\cos\theta\,d\phi$, with $d\sigma_\alpha=-\tfrac12\epsilon_{\alpha\beta\gamma}\sigma_\beta\wedge\sigma_\gamma$.

# Equations the tutorial uses

## (a) Meaning of "consistent Kaluza-Klein reduction"

Not given as a numbered equation; stated in prose. Introduction, PDF 2 (printed 1): a consistent reduction means a truncation of the complete Kaluza-Klein towers of massless and massive modes to the massless sector that holds "beyond the linearised level", i.e. at the full non-linear level; generically "currents built from the massless fields would act as sources for the massive fields that have been set to zero". Operationally (Sec. 3, PDF 12-13): one substitutes the ansatz into the D-dimensional equations of motion (18) and checks that "all the dependence on the internal $S^3$ coordinates $\mu^i$ has cancelled", leaving lower-dimensional equations of motion (21). Footnote 1 (PDF 4): the discussion concerns cases where *all* the SO(n+1) Yang-Mills fields of the sphere isometry group are retained, together with associated massless scalars. Footnote 13 (PDF 24-25) contrasts this with the "traditional group theory argument that proves conclusively the consistency of a truncation in which all singlets under a symmetry group are retained, and all non-singlets are truncated".

## (b) Main result: which (D, p, dilaton coupling) admit a consistent S^n reduction with full SO(n+1)

Group-theoretic criterion (Sec. 1 PDF 4, Sec. 2 PDF 6): the $T^n$ reduction must give a scalar coset $G/H$ with $H\supseteq SO(n+1)$; generic gravity + forms on $T^n$ gives only $GL(n,\mathbb R)$ with compact subgroup $SO(n)$, so "it is not possible to perform a consistent n-sphere reduction of a pure gravity theory, in which the Yang-Mills fields of SO(n+1) are retained" (PDF 4). Enhancement needs the p-form dilaton vectors to have the same length as the metric ones, i.e. $\delta = 4$. Without a dilaton ($\vec c=0$, $m=p$), eq. (2), PDF 8:
$$2D-4 = (p-1)(D-p-1),$$
whose only integer solutions are $(D,p)=(11,4),(11,7),(10,5)$. With a dilaton of coupling (4), reality of $a$ gives eq. (5): $p^2-3Dp+3D-5\ge 0$, so only $p=2,3,(D-2),(D-3)$ (the last two Hodge-dual). Resulting cosets after $T^n$ reduction: $p=3$: $\mathbb R\times O(n,n)/(O(n)\times O(n))$ for $D-n>3$ (eq. (8)), $O(D-2,D-2)/(O(D-2)\times O(D-2))$ for $D-n=3$ (eq. (9)); $p=2$: $GL(n+1,\mathbb R)/SO(n+1)$ (eq. (12)), since (3) with (6) is the $S^1$ reduction of pure gravity in $D+1$.

Table 1, PDF 11 (printed 10) — complete list (p-form; dilaton; higher dim; lower dim; sphere; gauge group; extra massless fields beyond metric, $A^{ij}_{(1)}$, $T_{ij}$):
- $F_{(2)}$, yes, any $D$, $D-2$, $S^2$, $SO(3)$, none.
- $F_{(3)}$, yes, any $D$, $D-3$, $S^3$, $SO(4)$, $A_{(2)}$.
- $F_{(3)}$, yes, any $D$, 3, $S^{D-3}$, $SO(D-2)$, none.
- $F_{(4)}$, no, 11, 7, $S^4$, $SO(5)$, $A^i_{(3)}$.
- $F_{(4)}$, no, 11, 4, $S^7$, $SO(8)$, $\phi_{[ijk\ell]_+}$.
- $F_{(5)}=*F_{(5)}$, no, 10, 5, $S^5$, $SO(6)$, none.

## (c) Reduction ansatz with SO(n+1) gauge fields

$S^3$ case, eq. (14), PDF 11-12 (printed 10-11), indices over 4 values:
$$d\hat s_D^2 = Y^{\frac{1}{D-2}}\Big(\Delta^{\frac{2}{D-2}}\,ds_{D-3}^2 + g^{-2}\Delta^{-\frac{D-4}{D-2}}\,T^{-1}_{ij}\,\mathcal{D}\mu^i\,\mathcal{D}\mu^j\Big),$$
$$e^{\sqrt{(D-2)/2}\,\hat\phi} = \Delta^{-1}Y^{(D-4)/4},$$
$$\hat F_{(3)} = F_{(3)} + \tfrac16\epsilon_{i_1i_2i_3i_4}\Big(g^{-2}U\Delta^{-2}\mathcal{D}\mu^{i_1}\wedge\mathcal{D}\mu^{i_2}\wedge\mathcal{D}\mu^{i_3}\mu^{i_4} - 3g^{-2}\Delta^{-2}\mathcal{D}\mu^{i_1}\wedge\mathcal{D}\mu^{i_2}\wedge\mathcal{D}T_{i_3j}\,T_{i_4k}\mu^j\mu^k - 3g^{-1}\Delta^{-1}F^{i_1i_2}_{(2)}\wedge\mathcal{D}\mu^{i_3}\,T_{i_4j}\mu^j\Big).$$
Its Hodge dual, eq. (20), PDF 13:
$$e^{-\sqrt{8/(D-2)}\hat\phi}\,\hat{*}\hat F_{(3)} = \tfrac{1}{6g^3}\epsilon_{ijk\ell}Y^{-1}{*F_{(3)}}\wedge\mu^i\mathcal{D}\mu^j\wedge\mathcal{D}\mu^k\wedge\mathcal{D}\mu^\ell - gU\epsilon_{D-3} + g^{-1}T^{-1}_{ij}{*\mathcal{D}T_{jk}}\wedge(\mu^k\mathcal{D}\mu^i) - \tfrac{1}{2g^2}T^{-1}_{ik}T^{-1}_{j\ell}{*F^{ij}_{(2)}}\wedge\mathcal{D}\mu^k\wedge\mathcal{D}\mu^\ell.$$
$S^{D-3}$ case, eq. (46), PDF 17 (printed 16), indices over $D-2$ values:
$$d\hat s_D^2 = Y^{\frac{1}{D-2}}\Big(\Delta^{\frac{D-4}{D-2}}ds_3^2 + g^{-2}\Delta^{-\frac{2}{D-2}}T^{-1}_{ij}\mathcal{D}\mu^i\mathcal{D}\mu^j\Big),\quad e^{-\sqrt{(D-2)/2}\hat\phi}=\Delta^{-1}Y^{1/2},$$
$$\hat F_{(3)} = -gU\epsilon_3 + g^{-1}T^{-1}_{ij}{*\mathcal{D}T_{jk}}\wedge(\mu^k\mathcal{D}\mu^i) - \tfrac{1}{2g^2}T^{-1}_{ik}T^{-1}_{j\ell}{*F^{ij}_{(2)}}\wedge\mathcal{D}\mu^k\wedge\mathcal{D}\mu^\ell.$$
$S^2$ case, eqs. (53)-(55), PDF 19 (printed 18), indices over 3 values:
$$d\hat s_D^2 = Y^{\frac{1}{D-2}}\Big(\Delta^{\frac{1}{D-2}}ds_{D-2}^2 + g^{-2}\Delta^{-\frac{D-3}{D-2}}T^{-1}_{ij}\mathcal{D}\mu^i\mathcal{D}\mu^j\Big),\quad e^{\sqrt{\frac{2(D-2)}{D-1}}\hat\phi}=\Delta^{-1}Y^{\frac{D-3}{D-1}},$$
$$\hat F_{(2)} = \tfrac12\epsilon_{ijk}\Big(g^{-1}U\Delta^{-2}\mu^i\mathcal{D}\mu^j\wedge\mathcal{D}\mu^k - 2g^{-1}\Delta^{-2}\mathcal{D}\mu^i\wedge\mathcal{D}T_{j\ell}T_{km}\mu^\ell\mu^m - \Delta^{-1}F^{ij}_{(2)}T_{k\ell}\mu^\ell\Big).$$
Lift of the $S^2$ case to $(D+1)$-dimensional pure gravity, eq. (60), PDF 20: $d\hat s_{D+1}^2 = Y^{\frac{2}{D-1}}ds^2_{D-2} + \Delta^{-1}Y^{\frac{2}{D-1}}T^{-1}_{ij}\mathcal{D}\mu^i\mathcal{D}\mu^j + \Delta Y^{-\frac{D-3}{D-1}}(dz+\hat A_{(1)})^2$ — an inhomogeneous deformation of a topological $S^3$ (Hopf fibration), not $S^1\times S^2$.

## (d) Reduced lower-dimensional Lagrangians

$S^3$, eq. (27)-(28), PDF 14 (printed 13):
$$\mathcal{L}_{D-3} = R\,{*1} - \tfrac{D-5}{16}Y^{-2}{*dY}\wedge dY - \tfrac14\tilde T^{-1}_{ij}{*\mathcal{D}\tilde T_{jk}}\wedge\tilde T^{-1}_{k\ell}\mathcal{D}\tilde T_{\ell i} - \tfrac12 Y^{-1}{*F_{(3)}}\wedge F_{(3)} - \tfrac14 Y^{-\frac12}\tilde T^{-1}_{ik}\tilde T^{-1}_{j\ell}{*F^{ij}_{(2)}}\wedge F^{k\ell}_{(2)} - V{*1},$$
$$V = \tfrac12 g^2 Y^{\frac12}\big(2\tilde T_{ij}\tilde T_{ij} - (\tilde T_{ii})^2\big),$$
with Chern-Simons-modified 3-form, eq. (29): $F_{(3)} = dA_{(2)} + \tfrac18\epsilon_{ijk\ell}\big(F^{ij}_{(2)}\wedge A^{k\ell}_{(1)} - \tfrac13 g A^{ij}_{(1)}\wedge A^{km}_{(1)}\wedge A^{m\ell}_{(1)}\big)$, satisfying the Bianchi identity (19): $dF_{(3)} = \tfrac18\epsilon_{i_1i_2i_3i_4}F^{i_1i_2}_{(2)}\wedge F^{i_3i_4}_{(2)}$.
$S^{D-3}$, eqs. (48)-(49), PDF 18 (printed 17): $\mathcal{L}_3 = R{*1} - \tfrac{1}{4(D-2)}Y^{-2}{*dY}\wedge dY - \tfrac14\tilde T^{-1}_{ij}{*\mathcal{D}\tilde T_{jk}}\wedge\tilde T^{-1}_{k\ell}\mathcal{D}\tilde T_{\ell i} - \tfrac14 Y^{-\frac{2}{D-2}}\tilde T^{-1}_{ik}\tilde T^{-1}_{j\ell}{*F^{ij}_{(2)}}\wedge F^{k\ell}_{(2)} - V{*1}$, $V=\tfrac12 g^2 Y^{\frac{2}{D-2}}(2\tilde T_{ij}\tilde T_{ij}-(\tilde T_{ii})^2)$. For $D=10$: $S^7$ reduction of the bosonic sector of $N=1$, $D=10$ supergravity gives $D=3$ $SO(8)$ gauged supergravity with 28 gauge fields and 36 scalars (35 + 1 of SO(8)); no AdS$_3$ vacuum, possibly half-SUSY domain walls (PDF 18-19).
$S^2$, eqs. (57)-(58), PDF 19-20: $\mathcal{L}_{D-2} = R{*1} - \tfrac{D-4}{3(D-1)}Y^{-2}{*dY}\wedge dY - \tfrac14\tilde T^{-1}_{ij}{*\mathcal{D}\tilde T_{jk}}\wedge\tilde T^{-1}_{k\ell}\mathcal{D}\tilde T_{\ell i} - \tfrac14 Y^{-\frac23}\tilde T^{-1}_{ik}\tilde T^{-1}_{j\ell}{*F^{ij}_{(2)}}\wedge F^{k\ell}_{(2)} - V{*1}$, $V=\tfrac12 g^2 Y^{\frac23}(2\tilde T_{ij}\tilde T_{ij}-(\tilde T_{ii})^2)$.
Intermediate $S^3$ equations of motion: eq. (21) (gauge-field, $\tilde T$, and $d(Y^{-1}*F_{(3)})=0$ equations, PDF 13) and eq. (26) for $Y$ (PDF 14).

## (e) Why generic reductions fail and the special cases work

- Generic failure (Intro, PDF 2): truncating to massless modes is inconsistent beyond linear order because currents built from massless fields source the massive fields set to zero; "one cannot make a consistent Kaluza-Klein n-sphere reduction of a generic theory in which all the massless fields, including, in particular, the full set of SO(n+1) gauge fields are retained".
- Necessary condition (PDF 4, PDF 6-8, PDF 24): gauging $SO(n+1)$ requires $SO(n+1)\subseteq H$ in the ungauged $T^n$ coset $G/H$; the generic $GL(n,\mathbb R)/O(n)$ is too small. Enhancement occurs only if the p-form dilaton vectors $\vec a_i$ have the same length as the metric ones $\vec b_i$ (which always have $\delta=4$), hence (2) with no dilaton and (4) with a dilaton. Footnote 2 (PDF 4): even for $(11,4)$ the $FFA$ term with the supersymmetric coefficient is needed ($E_7$), and for $(10,5)$ the 5-form must be (anti-)self-dual.
- Role of extra massless fields (PDF 5-6): in all cases the consistent reduction must keep *all* massless fields: $S^4$ needs the five 3-forms $A^i_{(3)}$, $S^7$ the 35 pseudoscalars $\phi_{[ijk\ell]_+}$, the new $S^3$ case the 2-form $A_{(2)}$, because "the Yang-Mills fields act as sources for them" [18]. The $S^5$, $S^{D-3}$ and $S^2$ cases need nothing extra.
- Status of the criterion (Sec. 8, PDF 24-25): it "does not, of itself, provide a guarantee of consistency. Rather, it can be viewed as providing a proof of inconsistency in cases where the necessary enhancement ... does not occur"; in every case where enhancement does occur, a consistent reduction was found. Sec. 7 illustrates: adding the conformal-anomaly cosmological term $-\tfrac12 m^2(D-26)e^{\frac12 a\hat\phi}\hat*1$ (eq. (61)) breaks the $\mathbb R$ factor but keeps $O(3,3)$, so the $S^3$ reduction stays consistent (potential (62): $V = \tfrac12 g^2Y^{\frac12}(2\tilde T_{ij}\tilde T_{ij}-(\tilde T_{ii})^2) + \tfrac12 m^2(D-26)Y^{\frac12}$), while the $S^{D-3}$ reduction is spoiled because $O(D-2,D-2)$ is broken (PDF 22). For $S^2$, a term $-\tfrac12 m^2 e^{b\hat\phi}\hat*1$ with $b^2 = \frac{2}{(D-1)(D-2)}$ (eqs. (66)-(67)) preserves consistency, potential (68): $V=\tfrac12 g^2Y^{\frac23}(\ldots)+\tfrac12 m^2 Y^{\frac{2}{D-2}}$, equivalent to $(D+1)$ gravity with cosmological constant (69).
- Einstein equation caveat (footnote 10, PDF 12): the D-dimensional Einstein equation was *not* checked in detail; consistency of the other equations plus agreement with known special cases is taken as evidence.
- Supersymmetry (Sec. 8, PDF 25-26): supersymmetry is "not in general playing a role" for the bosonic consistency; but once the bosonic massless sector is consistent, supersymmetry of the full theory guarantees that the fermionic sector can be consistently included (no massless-supercurrent-to-single-massive-field couplings), giving $S^3$ and $S^7$ reductions of $N=1$, $D=10$ supergravity.

## (f) S^2 and subgroup truncations

- $S^2$: Sec. 6, eqs. (50)-(60), PDF 19-20; summarised above in (c), (d). With a 2-form one could also consider $S^{D-2}$, "somewhat degenerate" since the lower theory would be 2-dimensional (PDF 10). Constant-scalar extremum with cosmological term: eq. (70) $\tilde T_{ij}=\delta_{ij}$, $Y^{\frac{2(D-5)}{D-2}} = \frac{m^2}{g^2(D-2)}$; eq. (71) $V = \tfrac12 m^2\frac{D-5}{D-2}\big[\frac{m^2}{g^2(D-5)}\big]^{\frac{1}{D-5}}$, positive for $D\ge6$ (de Sitter-type $M_{D-2}\times S^2$, lifting to $M_{D-2}\times S^3$ Hopf bundle).
- $SO(4)\to SU(2)$ (Sec. 4.1, PDF 14-16): self-dual truncation eq. (30) $A^{ij}_{(1)}=\pm\tfrac12\epsilon_{ijk\ell}A^{k\ell}_{(1)}$, scalars $T_{ij}=X\delta_{ij}$ (eq. (32), $Y=X^4$). Ansatz eqs. (35)-(38): $d\hat s_D^2 = X^{\frac{6}{D-2}}ds^2_{D-3} + \tfrac14 X^{-\frac{2(D-5)}{D-2}}\sum_\alpha(\sigma_\alpha - gA^\alpha_{(1)})^2$, $e^{\sqrt{(D-2)/2}\hat\phi}=X^{D-5}$, $\hat F_{(3)} = F_{(3)} - \tfrac{1}{4g^2}\Omega_{(3)} - \tfrac{1}{12g}\epsilon_{\alpha\beta\gamma}F^\alpha_{(2)}\wedge(\sigma_\beta-gA^\beta_{(1)})\wedge(\sigma_\gamma-gA^\gamma_{(1)})$. Consistency here is "no longer anything surprising": all retained fields are $SU(2)_R$ singlets, and they parametrise only *homogeneous* deformations of $S^3$ (footnote 7, PDF 9; PDF 16). Special case: $S^3$ reduction of $N=1$, $D=10$ to $D=7$ $SU(2)$ gauged supergravity [29].
- $SO(4)\to U(1)\times U(1)$ (Sec. 4.2, PDF 16-17): $A^{12}_{(1)}\equiv A^1_{(1)}$, $A^{34}_{(1)}\equiv A^2_{(1)}$, $\mu_1+i\mu_2=\tilde\mu_1e^{i\phi_1}$, $\mu_3+i\mu_4=\tilde\mu_2e^{i\phi_2}$, $T_{ij}=\mathrm{diag}(X_1,X_1,X_2,X_2)$ (eqs. (39)-(41)); metric eq. (43). Consistency is "still a non-trivial one" because $X_1,X_2$ parametrise inhomogeneous deformations; it follows from the full $SO(4)$ result. Reproduces [30] for $D=10$.

## (g) Intermediate equations of the S^3 reduction (for verification)

Lower-dimensional equations of motion read off from the ansatz, eq. (21), PDF 13 (printed 12):
$$(-1)^D\mathcal{D}\big(T^{-1}_{ik}T^{-1}_{j\ell}{*F^{k\ell}_{(2)}}\big) = -2g\,T^{-1}_{k[i}{*\mathcal{D}T_{j]k}} - \tfrac12\epsilon_{ijk\ell}Y^{-1}{*F_{(3)}}\wedge F^{k\ell}_{(2)},$$
$$(-1)^D\mathcal{D}\big(\tilde T^{-1}_{ik}{*\mathcal{D}\tilde T_{kj}}\big) = 2g^2\big[2T_{ik}T_{jk}-T_{ij}T_{kk}\big]\epsilon_{D-3} - T^{-1}_{\ell m}T^{-1}_{ik}{*F^{\ell k}_{(2)}}\wedge F^{mj}_{(2)} - \tfrac14\delta_{ij}\Big(2g^2\big[2T_{nk}T_{nk}-(T_{kk})^2\big]\epsilon_{D-3} - T^{-1}_{\ell m}T^{-1}_{nk}{*F^{\ell k}_{(2)}}\wedge F^{mn}_{(2)}\Big),$$
$$d\big(Y^{-1}{*F_{(3)}}\big)=0.$$
Dilaton from the ansatz, eq. (23), PDF 13: $d\hat\phi = \sqrt{\tfrac{2}{D-2}}\big(\tfrac14(D-4)Y^{-1}{*dY} - \Delta^{-1}d\Delta\big)$; the $\mu$-dependent piece needs the dual eq. (25): $\hat{*}(T_{ij}\mu^i\mathcal{D}\mu^j) = -\tfrac12\epsilon_{i_1i_2i_3i_4}T_{i\ell}\mu^i\epsilon_{D-3}\wedge(\Delta T_{i_1\ell}-T_{i_1j}T_{k\ell}\mu^j\mu^k)\mu^{i_2}\mathcal{D}\mu^{i_3}\wedge\mathcal{D}\mu^{i_4}$. The $Y$ equation, eq. (26), PDF 14:
$$\tfrac{D-5}{4}(-1)^D d\big(Y^{-1}{*dY}\big) = \tfrac12 g^2\big(2T_{ij}T_{ij}-(T_{ii})^2\big)\epsilon_{D-3} - Y^{-1}{*F_{(3)}}\wedge F_{(3)} - \tfrac14 Y^{-1}T^{-1}_{ik}T^{-1}_{j\ell}{*F^{ij}_{(2)}}\wedge F^{k\ell}_{(2)}.$$
The authors note the Schoutens identity $\epsilon_{[i_1i_2i_3i_4}V_{i_5]}\equiv0$ is needed for the cancellations (PDF 13). With the cosmological term (61), constant-scalar vacua of (62) require $V=0$, $\tilde T_{ij}=\delta_{ij}$ (eq. (64)) and $g = m\sqrt{(D-26)/8}$ (eq. (65), PDF 21), with $Y$ arbitrary: Minkowski, not AdS, ground state for $D>26$.

# Attributable statements

1. Generic sphere truncations to the massless sector fail beyond linear order because massless-field currents source the discarded massive modes (Sec. 1, PDF 2).
2. A consistent $S^n$ reduction retaining all $SO(n+1)$ gauge fields requires the $T^n$-reduced theory to have a scalar coset $G/H$ with $H\supseteq SO(n+1)$; generic $GL(n,\mathbb R)/O(n)$ is insufficient, so pure gravity never admits such a reduction (Sec. 1, PDF 4; Sec. 2, PDF 6).
3. For gravity plus a single p-form and no dilaton, symmetry enhancement — and hence a possible consistent sphere reduction — occurs only for $(D,p)=(11,4),(11,7),(10,5)$, eq. (2) (Sec. 2, PDF 8).
4. Adding a dilaton with coupling $a^2 = 4 - 2(p-1)(D-p-1)/(D-2)$ opens exactly $p=2$ ($S^2$) and $p=3$ ($S^3$ or $S^{D-3}$) in any dimension $D$, eqs. (4)-(7) and Table 1 (Sec. 2, PDF 8-11).
5. In every consistent case all massless fields must be kept: $A^i_{(3)}$ for $S^4$, $\phi_{[ijk\ell]_+}$ for $S^7$, $A_{(2)}$ for $S^3$; the Yang-Mills fields source them (Sec. 1, PDF 5-6; Table 1).
6. The explicit $S^3$ ansatz (14) with $\mathcal{D}\mu^i = d\mu^i + gA^{ij}_{(1)}\mu^j$ reduces (13) to the $(D-3)$-dimensional Lagrangian (27) with potential (28) (Sec. 3, PDF 11-14).
7. The $S^7$ reduction of $D=10$, $N=1$ supergravity gives a new $D=3$, $N=8$ $SO(8)$ gauged supergravity with 28 vectors and 36 scalars and no AdS$_3$ vacuum (Sec. 5, PDF 18-19).
8. The group-theoretic criterion proves inconsistency when enhancement fails but does not by itself guarantee consistency; the conformal-anomaly term kills the $S^{D-3}$ reduction but not the $S^3$ or $S^2$ ones (Sec. 7-8, PDF 21-25).
9. Truncating $SO(4)$ to $SU(2)_L$ makes consistency automatic by the singlet argument, and the retained fields are then only homogeneous deformations of $S^3$ (Sec. 4.1, PDF 14-16; footnote 7, PDF 9).
10. Once the bosonic massless sector is consistently reduced, higher-dimensional supersymmetry guarantees the fermions can be consistently included (Sec. 8, PDF 25-26).

# Caveats

- The D-dimensional Einstein equation is not verified in the paper (footnote 10, PDF 12); consistency is checked on the form-field and dilaton equations only. Later work (e.g. Cvetič-Lü-Pope-Sadrzadeh-Tran for $S^5$) addressed this; the tutorial should say "checked except for Einstein equation" if citing this paper alone.
- Signature and curvature conventions are never written down; the sign of the $Y$ kinetic term in (27) ($-\frac{D-5}{16}$) is the authors' and depends on them.
- In the printed eq. (14) the second term of $\hat F_{(3)}$ shows a plain "$D\mu^{i_1}$" where the gauge-covariant $\mathcal{D}\mu^{i_1}$ is presumably intended; transcribed above as $\mathcal{D}$. Likewise printed eq. (18) shows $\hat R_{MN} = \tfrac12\partial_M\phi\partial_N\phi + \tfrac14(\hat F^2_{MN} - \frac{2}{3(D-2)}\hat F^2_{(3)}\hat g_{MN})$ without the $e^{-a\hat\phi}$ factor and without hats on $\phi$ (compare eq. (52) for $p=2$, which does include $e^{-a\hat\phi}$); likely typographical.
- The text sometimes says "the full SO(4) reduction Ansatz of section 2" (PDF 15, 16) when the ansatz is in section 3.
- The paper asserts the completeness of Table 1 only within the class "gravity + one p-form + (possibly) one dilaton" with the full $SO(n+1)$ retained (footnote 1, PDF 4; footnote 5, PDF 7-8 excludes non-simply-laced enhancements such as $D=6$ self-dual 3-form $\to O(3,4)$ and $D=5$ 2-form $\to G_2$).
- The $S^2$ reduction can equally be seen as a reduction of $(D+1)$-dimensional pure gravity on a topological $S^3$ with the $U(1)$ fibre from the Dirac-monopole-like $\hat F_{(2)}$ (PDF 20); it is not an $S^1\times S^2$ reduction.
- "Consistent" throughout means consistency of the bosonic equations of motion; the full off-shell action-level statement and the fermionic sector are argued, not computed (Sec. 8).
- Equation (11) coset $GL(D-2,\mathbb R)/O(D-2)\ltimes V$ relies on dualising $(D-3)$ vectors to scalars in $D=3$ (footnote 8, PDF 10).
- Published venue not in the PDF; cite as arXiv:hep-th/0003286.
