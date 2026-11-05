# Identity

- arXiv: 0712.2717 [hep-th] (v1, 17 Dec 2007)
- Title: Coset Space Dimensional Reduction of Einstein-Yang-Mills theory
- Authors: A. Chatzistavrakidis, P. Manousselis, N. Prezas **and G. Zoupanos** (four authors; the title page lists Zoupanos as well)
- Year: 2007
- Venue: not stated in the PDF (header reads "preprint number hep-th/yymmnnn"); the abstract calls it a "contribution", i.e. a proceedings-style write-up extending ref. [12] = arXiv:0708.3222 (Phys. Lett. B 656 (2007) 152).
- Length: 15 PDF pages (title page unnumbered; printed page n = PDF page n+1).

# Structure

PDF page numbers first, printed page numbers in parentheses.

1. Title/abstract — PDF 1 (unnumbered)
2. Sec. 1 Introduction — PDF 2-3 (printed 1-2)
3. Sec. 2 Geometry of Coset spaces — PDF 3-5 (printed 2-4); eqs. (2.1)-(2.14)
4. Sec. 3 The Coset Space Dimensional Reduction — PDF 5-6 (printed 4-5); eqs. (3.1)-(3.6)
5. Sec. 4 Gravity and CSDR — PDF 6-9 (printed 5-8); eqs. (4.1)-(4.27)
6. Sec. 5 Reduction of the Gauge Sector: Gravity Modification of the CSDR Rules — PDF 9-12 (printed 8-11); eqs. (5.1)-(5.21)
7. Sec. 6 Conclusions — PDF 12 (printed 11)
8. Acknowledgements, References [1]-[24] — PDF 12-15 (printed 11-14)

# Conventions

- **Signature / curvature signs.** Not stated anywhere. The 4D tangent metric is written $\eta_{mn}$ (eq. 4.10) with no signature given. No explicit Riemann or Ricci sign convention; the curvature scalar is only defined implicitly through $\hat R *_D \mathbf 1 = \hat\Theta_{AB}\wedge *_D(\hat e^A\wedge\hat e^B)$ (4.22) with $\hat\Theta_{AB}$ the curvature 2-forms built from the spin connections (4.18)-(4.20). The Lagrangian (4.1) is $\hat R *\mathbf 1 - \tfrac12\mathrm{Tr}\hat F\wedge *\hat F - \hat\lambda *\mathbf 1$, i.e. $+R$ with Yang-Mills term $-\tfrac12 \mathrm{Tr} F\wedge *F$. No overall $1/16\pi G$ or $1/2\kappa^2$: the Einstein-Hilbert term is normalised to coefficient 1.
- **Coset $S/R$, dimension $d=\dim S/R$, $D = 4+d$.** Maurer-Cartan form $e(y)=L^{-1}(y)dL$ with $L(y^a)$ a coset representative, $a=1\ldots\dim S/R$ (Sec. 2, PDF 3). Index conventions: $A$ = $S$ (group) index, $a$ = coset index, $i$ = $R$ index. $e^a$ is the coframe, $e^i$ the $R$-connection. Coset coordinates $y^\alpha$ (curved internal index $\alpha$); 4D curved index $\mu$, 4D flat index $m,n,l$.
- **Structure constants.** Defined by $[Q_A,Q_B] = f_{AB}{}^C Q_C$ with the split (2.4); the index placement is first-two-lower, third-upper in commutators, and $f^A{}_{BC}$ (first up) in the Maurer-Cartan equation (2.2). Reductive coset assumed: $f_{bi}{}^j = 0$. For the reduced Lagrangian (4.26) they additionally assume the structure constants of $S$ are traceless and fully antisymmetric (PDF 9).
- **Normaliser split.** $N(R)/R$ generators $Q_{\hat a}$, remaining coset generators $Q_{\bar a}$; Lie algebra decomposition $S = R + K + L$ with $K \simeq \mathrm{Lie}(N(R)/R)$ and brackets (2.6). ($K$ here is a subalgebra of $S$; in Sec. 5 the same letter $K$ denotes the surviving 4D gauge group $\subset G$ — beware.)
- **Internal $S$-invariant metric (frozen background):** $g_{\alpha\beta}(y)=\delta_{ab}e^a_\alpha(y)e^b_\beta(y)$ (2.9); no radii appear in the frozen metric — the radii/moduli are introduced in Sec. 4 as $e^{2\beta\phi(x)}\gamma_{ab}(x)$ with $\gamma=\Phi^T\Phi$, $\det\Phi=1$.
- **Killing vectors.** Left isometries $S$: $K^\alpha_A = D^a_A e^\alpha_a$ with $D^B_A(s)$ in the adjoint of $S$ (2.13). Right isometries $N(R)/R$: $\tilde K^\alpha_{\hat a} = e^\alpha_{\hat a}$ (2.14), i.e. the inverse vielbein components along $\hat a$, with $e^{\hat a}_\alpha\tilde K^\alpha_{\hat b}=\delta^{\hat a}_{\hat b}$ (4.6).
- **Metric ansatz.** Kaluza-Klein form $d\hat s^2 = ds^2_{(4)} + h_{\alpha\beta}(dy^\alpha - \mathcal A^\alpha)(dy^\beta-\mathcal A^\beta)$ (4.2) with $\mathcal A^\alpha = \mathcal A^I(x)K^\alpha_{(I)}(y)$ (4.3); the *consistent* choice (4.4) keeps only $N(R)/R$ Killing vectors, $\mathcal A^\alpha = \mathcal A^{\hat a}(x)\tilde K^\alpha_{\hat a}(y)$. Full $S$-invariant ansatz (4.10) includes a breathing mode $\phi(x)$ with Weyl factors $e^{2\alpha\phi}$, $e^{2\beta\phi}$ and a unimodular moduli matrix $\gamma_{ab}(x)$. Notation follows Cvetic-Gibbons-Lu-Pope [13] (hep-th/0306043).
- **KK gauge field / field strength.** $\mathcal F^{\hat b} = d\mathcal A^{\hat b} + \tfrac12 f^{\hat b}{}_{\hat c\hat d}\mathcal A^{\hat c}\wedge\mathcal A^{\hat d}$ (4.9); no gauge coupling constant is introduced anywhere (all couplings absorbed, $g=1$). Gauge field of $G$: $\hat F = d\hat A + \hat A\wedge\hat A$ (below 4.1) and in components $\hat F^{\tilde I} = \hat d\hat A^{\tilde I} + \tfrac12 f^{\tilde I}{}_{\tilde J\tilde K}\hat A^{\tilde J}\wedge\hat A^{\tilde K}$ (5.2); $\tilde I,\tilde J,\tilde K$ are $G$ indices.
- **Hodge duals.** $*_D$, $*_4$, $*_d$ for $D$-, 4- and $d$-dimensional duals; $vol_4$, $vol_d$ the volume forms; identities (2.10)-(2.12) for the internal duals of 1-, 2-, 3-forms.
- **Higher-dimensional action (Lagrangian density $D$-form):**
  $$\mathcal L = \hat R *_D \mathbf 1 - \frac12 \mathrm{Tr}\,\hat F_{(2)}\wedge *_D\hat F_{(2)} - \hat\lambda_{(D)} *_D\mathbf 1 \quad (4.1,\ \text{PDF 6})$$
  with $\hat\lambda_{(D)}$ the $D$-dimensional cosmological constant.

# Equations the tutorial uses

## (f) Lie-algebra data for the cosets (Sec. 2, PDF 3-5)

$$e(y)=e^A Q_A = e^a Q_a + e^i Q_i \quad (2.1)$$
$$de^A = -\frac12 f^A{}_{BC}\,e^B\wedge e^C \quad (2.2)$$
$$de^a = -\frac12 f^a{}_{bc}\,e^b\wedge e^c - f^a{}_{bi}\,e^b\wedge e^i,\qquad de^i = -\frac12 f^i{}_{ab}\,e^a\wedge e^b - \frac12 f^i{}_{jk}\,e^j\wedge e^k \quad (2.3)$$
$$[Q_i,Q_j]=f_{ij}{}^k Q_k,\quad [Q_i,Q_a]=f_{ia}{}^b Q_b,\quad [Q_a,Q_b]=f_{ab}{}^c Q_c + f_{ab}{}^i Q_i \quad (2.4)$$
Reductive: $f_{bi}{}^j=0$. Normaliser $N=\{s\in S,\ sRs^{-1}\subset R\}$ (2.5); $S=R+K+L$ with
$$[K,K]\subset K,\quad [K,R]=0,\quad [K,L]\subset L,\quad [L,R]\subset L,\quad [L,L]=L+R \quad (2.6)$$
Split of (2.4) with $Q_a\to(Q_{\hat a},Q_{\bar a})$:
$$[Q_{\hat a},Q_{\hat b}]=f_{\hat a\hat b}{}^{\hat c}Q_{\hat c},\quad [Q_i,Q_{\hat a}]=0,\quad [Q_{\hat a},Q_{\bar a}]=f_{\hat a\bar a}{}^{\bar b}Q_{\bar b},\quad [Q_i,Q_{\bar a}]=f_{i\bar a}{}^{\bar b}Q_{\bar b},\quad [Q_{\bar a},Q_{\bar b}]=f_{\bar a\bar b}{}^{\bar c}Q_{\bar c}+f_{\bar a\bar b}{}^iQ_i \quad (2.7)$$
$$de^{\hat a}=-\frac12 f^{\hat a}{}_{\bar b\bar c}e^{\bar b}\wedge e^{\bar c},\quad de^{\bar a}=-\frac12 f^{\bar a}{}_{\bar b\bar c}e^{\bar b}\wedge e^{\bar c}-f^{\bar a}{}_{\hat b\bar c}e^{\hat b}\wedge e^{\bar c}-f^{\bar a}{}_{\bar b i}e^{\bar b}\wedge e^i,\quad de^i=-\frac12 f^i{}_{\bar b\bar c}e^{\bar b}\wedge e^{\bar c}-\frac12 f^i{}_{jk}e^j\wedge e^k \quad (2.8)$$
$$g_{\alpha\beta}(y)=\delta_{ab}e^a_\alpha(y)e^b_\beta(y) \quad (2.9)$$
$$e^a\wedge *_d e^b=\delta^{ab}vol_d,\quad (e^a\wedge e^b)\wedge *_d(e^c\wedge e^d)=\delta^{ab}_{cd}vol_d,\quad (e^a\wedge e^b\wedge e^c)\wedge *_d(e^d\wedge e^e\wedge e^f)=\delta^{abc}_{def}vol_d \quad (2.10\text{-}2.12)$$
$$K^\alpha_A = D^a_A e^\alpha_a \quad (2.13),\qquad \tilde K^\alpha_{\hat a}=e^\alpha_{\hat a} \quad (2.14)$$
No explicit structure constants for any particular $S$ (e.g. SU(2)) are given.

## (c) CSDR of the Yang-Mills sector — frozen-metric reminder (Sec. 3, PDF 5-6)

$$\mathcal L_{X^I}\hat A = DW_I \quad (3.1),\qquad \mathcal L_{X^I}g_{MN}=0 \quad (3.2)$$
$$\mathcal L_{X^I}\hat A = i_{X^I}d\hat A + d\,i_{X^I}\hat A = DW_I = dW_I + [\hat A,W_I] \quad (3.3),\qquad [\mathcal L_{X^I},\mathcal L_{X^J}]=\mathcal L_{[X^I,X^J]} \quad (3.4)$$
Scherk-Schwarz-like ansatz shown equivalent to CSDR in [12]:
$$\hat A^{\tilde I}(x,y)=A^{\tilde I}(x)+\chi^{\tilde I}_\alpha(x,y)dy^\alpha \quad (3.5),\qquad \chi^{\tilde I}_\alpha(x,y)=\phi^{\tilde I}_A(x)e^A_\alpha(y) \quad (3.6)$$
$\phi_A(x)$ take values in Lie($G$) and are the 4D Higgs fields.

## (a) Metric ansatz and reduction of the Einstein-Hilbert term (Sec. 4, PDF 6-8)

$$d\hat s^2_{(D)}=ds^2_{(4)}+h_{\alpha\beta}(x,y)(dy^\alpha-\mathcal A^\alpha(x,y))(dy^\beta-\mathcal A^\beta(x,y)) \quad (4.2)$$
$$\mathcal A^\alpha(x,y)=\mathcal A^I(x)K^\alpha_{(I)}(y),\quad \mathcal A^I(x)=\mathcal A^I_\mu(x)dx^\mu \quad (4.3)$$
Consistent truncation (refs [21],[16]): only $N(R)/R$ Killing vectors,
$$\mathcal A^\alpha(x,y)=\mathcal A^{\hat a}(x)\tilde K^\alpha_{\hat a}(y) \quad (4.4)$$
$$\eta^{\hat a}=e^{\hat a}_\alpha(dy^\alpha-A^{\hat b}(x)\tilde K^\alpha_{\hat b}(y))=e^{\hat a}-A^{\hat a}(x) \quad (4.5),\qquad e^{\hat a}_\alpha\tilde K^\alpha_{\hat b}=\delta^{\hat a}_{\hat b} \quad (4.6)$$
$$\eta^{\bar a}=e^{\bar a},\qquad e^i=e^i_a e^a \quad (4.7)$$
$$D\eta^{\hat a}\equiv d\eta^{\hat a}+f^{\hat a}{}_{\hat b\hat c}\mathcal A^{\hat b}\wedge\eta^{\hat c}=-\mathcal F^{\hat a}-\frac12 f^{\hat a}{}_{\hat b\hat c}\eta^{\hat b}\wedge\eta^{\hat c} \quad (4.8)$$
$$\mathcal F^{\hat b}=d\mathcal A^{\hat b}+\frac12 f^{\hat b}{}_{\hat c\hat d}\mathcal A^{\hat c}\wedge\mathcal A^{\hat d} \quad (4.9)$$
General $S$-invariant metric ansatz:
$$d\hat s^2_{(D)}=e^{2\alpha\phi(x)}\eta_{mn}e^m e^n+e^{2\beta\phi(x)}\gamma_{ab}(x)\eta^a\eta^b \quad (4.10)$$
$$\hat e^m=e^{\alpha\phi}e^m,\qquad \hat e^a=e^{\beta\phi}\Phi^a_b(x)\eta^b \quad (4.11),\qquad \gamma_{cd}(x)=\delta_{ab}\Phi^a_c(x)\Phi^b_d(x) \quad (4.12),\qquad (\Phi^{-1})^c_a(\Phi^{-1})^d_b\gamma_{cd}=\delta_{ab} \quad (4.13)$$
($\Phi$ has unit determinant.) Exterior derivatives of the vielbeins:
$$d\hat e^m=-\omega^m{}_n\wedge\hat e^n-\alpha e^{-\alpha\phi}\partial_\nu\phi\,\hat e^m\wedge\hat e^n \quad (4.14)\ \text{[sic: }\partial_\nu\text{ printed, }\partial_n\text{ evidently meant]}$$
$$d\hat e^a=-\tilde f^a{}_{ib}\,e^i\wedge\hat e^b+e^{-\alpha\phi}D_{bn}{}^a\hat e^{nb}+\beta e^{-\alpha\phi}\partial_m\phi\,\hat e^{ma}-\frac12 e^{(\beta-2\alpha)\phi}\mathcal F^a{}_{mn}\hat e^{mn}-\frac12\tilde f^a{}_{bc}\hat e^{bc} \quad (4.15)$$
$$e^{ab}\equiv e^a\wedge e^b,\qquad \mathcal F^a_{mn}\equiv\Phi^a_{\hat a}\mathcal F^{\hat a}_{mn} \quad (4.16)$$
$$\tilde f^a{}_{ib}=\Phi^a_c(\Phi^{-1})^d_b f^c{}_{id},\quad D_{bn}{}^a=(\Phi^{-1})^c_b D_n\Phi^a_c,\quad \tilde f^a{}_{bc}=\Phi^a_d(\Phi^{-1})^e_b(\Phi^{-1})^f_c f^d{}_{ef} \quad (4.17)$$
Spin connections:
$$\hat\omega_{mn}=\omega_{mn}+\frac12 e^{(\beta-2\alpha)\phi}\mathcal F^a{}_{mn}\hat e^a+\alpha e^{-\alpha\phi}(\partial_n\phi\,\eta_{ml}\hat e^l-\partial_m\phi\,\eta_{nl}\hat e^l) \quad (4.18)$$
$$\hat\omega_{ma}=-e^{-\alpha\phi}P_{mab}\hat e^b-\beta e^{-\alpha\phi}\partial_m\phi\,\hat e^a+\frac12 e^{(\beta-2\alpha)\phi}\mathcal F_{aml}\hat e^l \quad (4.19)$$
$$\hat\omega_{ab}=-\tilde f_{iab}e^i-e^{-\alpha\phi}Q_{mab}\hat e^m+e^{-\beta\phi}\tilde C_{cab}\hat e^c \quad (4.20)$$
$$\tilde C_{cab}=\frac12(\tilde f^c{}_{ab}+\tilde f^b{}_{ac}-\tilde f^a{}_{bc}),\quad P_{mab}=\frac12[(\Phi^{-1})^c_a D_m\Phi^b_c+(\Phi^{-1})^c_b D_m\Phi^a_c],\quad Q_{mab}=\frac12[(\Phi^{-1})^c_a D_m\Phi^b_c-(\Phi^{-1})^c_b D_m\Phi^a_c],\quad D_m\Phi^a_d=\partial_m\Phi^a_d-f^c{}_{d\hat b}\mathcal A^{\hat b}_m\Phi^a_c \quad (4.21)$$
Curvature scalar (only given implicitly; no explicit $\hat R$ of the ansatz is written out):
$$\hat R*_D\mathbf 1=\hat\Theta_{AB}\wedge *_D(\hat e^A\wedge\hat e^B),\quad A=m,a \quad (4.22)$$
Constraints required for the reduction to go through (satisfied by Jacobi identities and invariance of the metric):
$$-\frac12\tilde f^a{}_{ib}f^i{}_{jk}+\tilde f^a{}_{jc}\tilde f^c{}_{kb}=0,\qquad -\tilde C^a_{cb}\tilde f^c{}_{id}+\tilde C^a_{dc}\tilde f^c{}_{ib}-\tilde C^c_{db}\tilde f^a{}_{ic}=0 \quad (4.23)$$

## (b) Reduced gravitational action (Sec. 4, PDF 9)

$$\begin{aligned}\mathcal L=e^{4\alpha+d\beta}\Big(&e^{-2\alpha\phi}R*\mathbf 1-e^{-2\alpha\phi}*P_{ab}\wedge P_{ab}-\frac12 e^{2(\beta-2\alpha)\phi}\gamma_{ab}*\mathcal F^a\wedge\mathcal F^b\\&+e^{-2\alpha\phi}\big((3\alpha+d\beta)^2-(3\alpha^2+d\beta^2)\big)*d\phi\wedge d\phi\\&-\frac14 e^{-2\beta\phi}\big(\gamma_{ab}\gamma^{cd}\gamma^{ef}f^a{}_{ce}f^b{}_{df}+2\gamma^{ab}f^c{}_{da}f^d{}_{cb}+4e^{2\beta\phi}\gamma^{ab}f_{iac}f_b{}^{ic}\big)*\mathbf 1+\lambda_D*\mathbf 1\Big)\end{aligned} \quad (4.24)$$
[sic: prefactor printed as $e^{4\alpha+d\beta}$; $e^{(4\alpha+d\beta)\phi}$ is evidently meant.]
$$\alpha=-\sqrt{\frac{d}{4d+8}},\qquad \beta=-\frac{2\alpha}{d} \quad (4.25)$$
With $\det\Phi=1$ and traceless, fully antisymmetric $f$:
$$\mathcal L=R*\mathbf 1-*P_{ab}\wedge P_{ab}-\frac12 e^{2(\beta-\alpha)\phi}\gamma_{\hat a\hat b}*\mathcal F^{\hat a}\wedge\mathcal F^{\hat b}-\frac12*d\phi\wedge d\phi-V(\phi) \quad (4.26)$$
$$V=\frac14 e^{2(\alpha-\beta)\phi}\big(\gamma_{ab}\gamma^{cd}\gamma^{ef}f^a{}_{ce}f^b{}_{df}+2\gamma^{ab}f^c{}_{da}f^d{}_{cb}+4e^{2\beta\phi}\gamma^{ab}f_{iac}f_b{}^{ic}-4e^{2\beta\phi}\lambda_D\big)*\mathbf 1 \quad (4.27)$$
"The first two terms in (4.27) have a non-zero contribution only in the case of non-symmetric coset spaces" (PDF 9). Internal check (mine): $\beta=-2\alpha/d$ cancels the $R$ prefactor; $(3\alpha+d\beta)^2-3\alpha^2-d\beta^2=-2\alpha^2(d+2)/d=-\tfrac12$ exactly for $\alpha^2=d/(4d+8)$; and $(4\alpha+d\beta)+2(\beta-2\alpha)=2(\beta-\alpha)$, $(4\alpha+d\beta)-2\beta=2(\alpha-\beta)$, so (4.26)-(4.27) follow from (4.24)-(4.25) as printed.

## (c) CSDR of the Yang-Mills sector with dynamical gravity (Sec. 5, PDF 9-12)

$$\hat A^{\tilde I}=A^{\tilde I}+\phi^{\tilde I}_A\eta^A,\qquad \eta^{\hat a}=e^{\hat a}-\mathcal A^{\hat a},\ \eta^{\bar a}=e^{\bar a},\ \eta^i=e^i=e^i_a e^a \quad (5.1)$$
$$\hat F=\hat d\hat A^{\tilde I}+\frac12 f^{\tilde I}{}_{\tilde J\tilde K}\hat A^{\tilde J}\wedge\hat A^{\tilde K} \quad (5.2)$$
$$\hat F^{\tilde I}=(F^{\tilde I}-\mathcal F^{\hat a}\phi^{\tilde I}_{\hat a})+D\phi^{\tilde I}_A\wedge\eta^A-\frac12 F^{\tilde I}_{AB}\eta^A\wedge\eta^B \quad (5.3)$$
$$F^{\tilde I}=dA^{\tilde I}+\frac12 f^{\tilde I}{}_{\tilde J\tilde K}A^{\tilde J}\wedge A^{\tilde K} \quad (5.4),\qquad F^{\tilde I}_{AB}=f^C{}_{AB}\phi^{\tilde I}_C-f^{\tilde I}{}_{\tilde J\tilde K}\phi^{\tilde J}_A\phi^{\tilde K}_B \quad (5.5)$$
$$D\phi^{\tilde I}_A=d\phi^{\tilde I}_A+f^C{}_{AB}\mathcal A^B\phi^{\tilde I}_C+f^{\tilde I}{}_{\tilde J\tilde K}A^{\tilde J}\phi^{\tilde K}_A \quad (5.6)$$
$$*_D\hat F^{\tilde I}=*_4(F^{\tilde I}-\mathcal F^{\hat a}\phi^{\tilde I}_{\hat a})\wedge vol_d+e^{\alpha\phi-\beta\phi}*_4D\phi^{\tilde I}_A\wedge *_d\tilde\eta^A-\frac12 e^{2\alpha\phi-2\beta\phi}F^{\tilde I}_{AB}vol_4\wedge *_d(\tilde\eta^A\wedge\tilde\eta^B) \quad (5.7)$$
Inserted in $\mathcal L=-\frac12\mathrm{Tr}\hat F\wedge *_D\hat F$:
$$\begin{aligned}\mathcal L=&-\frac12 e^{-2\alpha\phi}(F^{\tilde I}-\mathcal F^{\hat a}\phi^{\tilde I}_{\hat a})\wedge *_4(F^{\tilde I}-\mathcal F^{\hat a}\phi^{\tilde I}_{\hat a})\wedge vol_d-\frac12 e^{-2\beta\phi}D\phi^{\tilde I}_A\wedge *_4D\phi^{\tilde I}_B\wedge\tilde\eta^A\wedge *_d\tilde\eta^B\\&+\frac14 e^{2\alpha\phi-4\beta\phi}F_{AB}F_{CD}vol_4\wedge\tilde\eta^A\wedge\tilde\eta^B\wedge *_d(\tilde\eta^C\wedge\tilde\eta^D)\end{aligned} \quad (5.8)$$
$$\tilde\eta^a=(\Phi^{-1})^a_b\eta^b,\qquad \tilde\eta^i=e^i_a(\Phi^{-1})^a_b\eta^b \quad (5.9)$$
Constraints: usual CSDR set
$$D\phi^I_i=0,\qquad F^I_{ij}=F^I_{aj}=0 \quad (5.10)$$
and gravity-induced set
$$D\phi_{\hat a}=F_{\hat a\hat b}=F_{\hat a\bar b}=F_{\hat a i}=0 \quad (5.11)$$
From $F^{\tilde I}_{\hat a\hat b}=f^{\hat c}{}_{\hat a\hat b}\phi^{\tilde I}_{\hat c}-[\phi_{\hat a},\phi_{\hat b}]^{\tilde I}=0$: $\phi_{\hat a}$ are generators of an $N(R)/R$ subgroup of $H$ ($H$ = centraliser of $R$ in $G$), and
$$f^{\hat c}{}_{\hat a\hat b}\phi^{\tilde I}_{\hat c}=f^{\tilde I}{}_{\tilde J\tilde K}\phi^{\tilde J}_{\hat a}\phi^{\tilde K}_{\hat b} \quad (5.12)$$
Then $D\phi^{\tilde I}_{\hat a}=0$ with $\phi_{\hat a}$ constant gives (as printed, free indices do not match — $\hat a$ in first term, $\hat b$ in second)
$$f^{\tilde I}{}_{\tilde J\tilde K}\phi^{\tilde J}_{\hat a}\phi^{\tilde K}_{\hat b}\mathcal A^{\hat b}+f^{\tilde I}{}_{\tilde J\tilde K}A^{\tilde J}\phi^{\tilde K}_{\hat b}=0 \quad (5.13)$$
Solution: the $N(R)/R$ part of the $H$ gauge field is locked to the KK gauge field,
$$A^{\tilde I}=\mathcal A^{\hat b}\phi^{\tilde I}_{\hat b} \quad (5.14),\qquad F^{\tilde I}=d\mathcal A^{\hat a}\phi^{\tilde I}_{\hat a}+\frac12 f^{\hat c}{}_{\hat a\hat b}\mathcal A^{\hat a}\wedge\mathcal A^{\hat b}\phi^{\tilde I}_{\hat c}=\mathcal F^{\hat a}\phi^{\tilde I}_{\hat a} \quad (5.15)$$
which is exactly the term subtracted in (5.3). Surviving gauge group $K$ from $G\supset R\times H$, $H\supset(N(R)/R)\times K$. $F_{\hat a i}=[\phi_{\hat a},\phi_i]=0$ trivially; scalar representations fixed by
$$F_{\bar a\bar b}=f^{\bar c}{}_{\bar a\bar b}\phi_{\bar c}-[\phi_{\bar a},\phi_{\bar b}]=0 \quad (5.16),\qquad F_{\bar a i}=f^{\bar c}{}_{\bar a i}\phi_{\bar c}-[\phi_{\bar a},\phi_i]=0 \quad (5.17)$$
$$S\supset R\times(N(R)/R),\quad adS=adR+adN(R)/R+\sum(r_i,n_i) \quad (5.18)$$
$$G\supset R\times(N(R)/R)\times K,\quad adG=(adR,1,1)+(1,adN(R)/R,1)+(1,1,adK)+\sum(l_i,m_i,k_i) \quad (5.19)$$
A $k_i$ multiplet of scalars survives when $(r_i,n_i)=(l_i,m_i)$.

## (d) Total 4D gauge-sector Lagrangian and potential (PDF 12)

$$\mathcal L=-\frac12 e^{-2\alpha\phi}F^{\tilde I}\wedge *_4F^{\tilde I}\wedge vol_d-\frac12 e^{-2\beta\phi}\gamma^{\bar a\bar b}D\phi^{\tilde I}_{\bar a}\wedge *_4D\phi^{\tilde I}_{\bar b}\wedge vol_d+\frac14 e^{2\alpha\phi-4\beta\phi}\gamma^{\bar a\bar c}\gamma^{\bar b\bar d}F_{\bar a\bar b}F_{\bar c\bar d}\,vol_4\wedge vol_d \quad (5.20)$$
$$V_{gt}=-\frac14 e^{(2\alpha\phi-4\beta\phi)}\gamma^{\bar a\bar c}\gamma^{\bar b\bar d}F_{\bar a\bar b}F_{\bar c\bar d} \quad (5.21)$$
The paper never writes a single combined "total" 4D action: the gravity sector is (4.26)-(4.27) and the gauge sector is (5.20)-(5.21); the tutorial must add them itself (and note that (5.20) still carries $vol_d$ and no Weyl-rescaling prefactor from (4.24), see Caveats).

## (e) Explicit examples

None. No $S^2=SU(2)/U(1)$ or any other explicit coset is worked out; the only concrete statement is that the $f^a{}_{ce}f^b{}_{df}$ and $f^c{}_{da}f^d{}_{cb}$ terms of $V$ vanish for symmetric cosets (PDF 9) and that $R$ has no $N(R)/R$ subgroup (PDF 10).

# Attributable statements

1. (Sec. 1, PDF 2) The main obstacle for realistic pure Kaluza-Klein models is the absence of chiral fermions in 4D [Witten]; adding Yang-Mills fields in higher dimensions cures this via the Atiyah-Hirzebruch theorem and allows spontaneous compactification.
2. (Sec. 1, PDF 3) Reduction over CY spaces gives supersymmetric 4D theories, over symmetric cosets non-supersymmetric ones, over non-symmetric cosets softly broken supersymmetry.
3. (Sec. 2, PDF 4, below 2.4) The coset is assumed reductive, $f_{bi}{}^j=0$, following Castellani-Romans-Warner [14].
4. (Sec. 2, PDF 5, eqs. 2.13-2.14) $S/R$ has a left isometry $S$ with Killing vectors $K^\alpha_A=D^a_Ae^\alpha_a$ and a right isometry $N(R)/R$ with Killing vectors $\tilde K^\alpha_{\hat a}=e^\alpha_{\hat a}$.
5. (Sec. 3, PDF 6, eqs. 3.5-3.6) The Scherk-Schwarz-type ansatz $\hat A=A(x)+\phi_A(x)e^A$ was shown in [12] (arXiv:0708.3222) to be equivalent to the CSDR symmetric-field ansatz.
6. (Sec. 4, PDF 6) "A well known problem with coset reductions is that we cannot consistently allow Kaluza-Klein gauge fields from the full isometry group $S$ of the coset $S/R$ to survive"; following Coquereaux-Jadczyk [21] and Cho-Kim [16] the consistent truncation keeps only $N(R)/R$ KK gauge fields (4.4).
7. (Sec. 4, PDF 8, eq. 4.23) The reduction of the Einstein-Hilbert term requires constraints (4.23), which are satisfied by the Jacobi identities and the invariance of the metric.
8. (Sec. 4, PDF 9, eq. 4.25) The Einstein-frame / canonical choice is $\alpha=-\sqrt{d/(4d+8)}$, $\beta=-2\alpha/d$.
9. (Sec. 4, PDF 9) The first two terms of the moduli potential (4.27) are non-zero only for non-symmetric coset spaces.
10. (Sec. 5, PDF 11, eqs. 5.14-5.15) With dynamical gravity, the $N(R)/R$ part of the $H$ gauge field is identified with the KK gauge field, $A^{\tilde I}=\mathcal A^{\hat b}\phi^{\tilde I}_{\hat b}$, so the surviving 4D gauge group is $K$ with $H\supset(N(R)/R)\times K$; (Sec. 6, PDF 12) "the emergence of the full isometry of the coset as a part of the four-dimensional gauge group is not permitted."

# Caveats

- **Which modes are kept.** This is a consistent-truncation (Scherk-Schwarz/CSDR-type) ansatz, not a harmonic expansion. Kept: 4D metric, the breathing mode $\phi(x)$, the unimodular moduli $\gamma_{ab}(x)=(\Phi^T\Phi)_{ab}$ (with $\det\Phi=1$, these are the "radii"/squashing modes), KK gauge fields $\mathcal A^{\hat a}$ for $N(R)/R$ only (not all of $S$), $K$ gauge fields $A^{\tilde I}$ and Higgs scalars $\phi_{\bar a}$. The $S$ isometry does NOT yield 4D gauge bosons; this is the main message and differs from naive KK lore.
- **Consistency of (4.4).** Justified by citing [21] Coquereaux-Jadczyk and [16] Cho-Kim; the paper does not itself prove consistency of the truncation, and does not discuss the $P_{ab}$/$\gamma_{ab}$ moduli as a consistent truncation beyond stating the constraints (4.23).
- **The Einstein-Hilbert reduction is only sketched.** (4.22) names the curvature 2-forms $\hat\Theta_{AB}$ but never writes them or $\hat R$ explicitly; the jump from (4.18)-(4.21) to (4.24) is not shown. A tutorial wanting to verify (4.24) must compute $\hat\Theta_{AB}$ from the given spin connections itself.
- **Typos / index mismatches to be aware of:** (4.14) has $\partial_\nu\phi$ where $\partial_n\phi$ is meant; (4.24) prefactor $e^{4\alpha+d\beta}$ lacks the $\phi$; (5.13) has a free $\hat a$ in one term and $\hat b$ in the other; in (4.21) $D_m\Phi^a_d$ uses $f^c{}_{d\hat b}$ with mixed index positions; in (4.24) $\gamma_{ab}*\mathcal F^a\wedge\mathcal F^b$ runs over all $a$ but (4.26) correctly restricts to $\hat a\hat b$ via (4.16). In (4.5) the KK field appears as $A^{\hat b}$ but elsewhere as $\mathcal A^{\hat b}$.
- **Cosmological constant sign.** (4.1) has $-\hat\lambda_{(D)}*\mathbf 1$ but (4.24) has $+\lambda_D*\mathbf 1$ inside the bracket and (4.27) has $-4e^{2\beta\phi}\lambda_D$ inside $V$ (hence $+\lambda_D$ in $\mathcal L$). Sign is not consistent between (4.1) and (4.24)/(4.27); check before quoting.
- **Scalar kinetic normalisation.** The $(3\alpha+d\beta)^2-(3\alpha^2+d\beta^2)$ coefficient and (4.25) are mutually consistent (yield $-\tfrac12 *d\phi\wedge d\phi$) as verified above, but the origin of "3" (rather than something tied to $D-2$) is not explained; compare with Cvetic-Gibbons-Lu-Pope [13] before reusing.
- **Gauge sector not Weyl-rescaled.** (5.8) and (5.20) carry the factors $e^{-2\alpha\phi}$, $e^{-2\beta\phi}$, $e^{2\alpha\phi-4\beta\phi}$ and $vol_d$ from the $D$-dimensional Hodge dual but no $e^{(4\alpha+d\beta)\phi}$ determinant prefactor as in (4.24); so (5.20) is not yet in the same frame/normalisation as (4.26). Combining the two sectors requires redoing this consistently.
- **No gauge coupling, no $\kappa$.** All couplings are set to one and no internal volume factor is extracted; the tutorial has to reinstate $1/2\kappa^2$, $1/g^2$ and $\int vol_d$ by hand.
- **Index placement of structure constants** is not uniform: $f_{AB}{}^C$ in (2.4) vs $f^A{}_{BC}$ in (2.2), and fully lowered $f_{iac}$, $f_b{}^{ic}$ in (4.24); the authors assume full antisymmetry (PDF 9) so this is harmless only with a Killing-metric-normalised basis.
- **Letter clash.** $K$ = subalgebra of $S$ isomorphic to Lie$(N(R)/R)$ in Sec. 2, but $K$ = surviving 4D gauge group $\subset H\subset G$ in Sec. 5.

# Sections and equations used (part-iii-02-gauge-tower, sec-kk-fermions; task 12.7)

- Sec. 1, PDF p. 2 (absence of chiral fermions in pure Kaluza-Klein models, cured by higher-dimensional Yang-Mills fields): exm-kkf-standard-model.
