# Figure: two-dimensional de Sitter space as the hyperboloid X0^2 - X1^2 - X2^2 = -1/H^2 in 2+1 Minkowski space
# (signature (+, -, -)),
# with the flat-slicing coordinates: lines of constant t (circles growing like e^{Ht}) and of constant x.
# Output: figures/de_sitter.png.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "de_sitter.png")
t, x, H = var("t x H"); Hv = 1
# flat slicing of dS_2: X0 = sinh(Ht) + (H^2 x^2/2) e^{Ht}, X1 = x e^{Ht}, X2 = cosh(Ht) - (H^2 x^2/2) e^{Ht}  (times 1/H)
sh = (exp(Hv*t) - exp(-Hv*t))/2; ch = (exp(Hv*t) + exp(-Hv*t))/2       # sinh, cosh written with exponentials
X0 = (sh + Hv^2*x^2/2*exp(Hv*t))/Hv; X1 = x*exp(Hv*t); X2 = (ch - Hv^2*x^2/2*exp(Hv*t))/Hv
scene = parametric_plot3d((X1, X2, X0), (t, -1.5, 1.2), (x, -2.5, 2.5), color="lightsteelblue", opacity=0.6, mesh=False)
for tk in [-1, -0.5, 0, 0.5, 1]:
    scene += parametric_plot3d([e.subs(t=tk) for e in (X1, X2, X0)], (x, -2.5, 2.5), color="firebrick", thickness=2)
for xk in [-2, -1, 0, 1, 2]:
    scene += parametric_plot3d([e.subs(x=xk) for e in (X1, X2, X0)], (t, -1.5, 1.2), color="royalblue", thickness=1.5)
scene.save(OUT, viewer="tachyon", figsize=[7, 7], zoom=1.1, aspect_ratio=1)
print("[PASS] the embedding lies on the hyperboloid X0^2 - X1^2 - X2^2 = -1/H^2:", bool((X0^2 - X1^2 - X2^2 + 1/Hv^2).simplify_full() == 0))
# induced metric in (t, x): dt^2 - e^{2Ht} dx^2
E = vector([X0, X1, X2]); J = jacobian(E, (t, x)); eta = diagonal_matrix([1, -1, -1])
G = (J.transpose()*eta*J).simplify_full()
print("[PASS] induced metric is dt^2 - e^{2Ht} dx^2:", bool(G == matrix([[1, 0], [0, -exp(2*Hv*t)]])))
print("wrote", os.path.relpath(OUT))
