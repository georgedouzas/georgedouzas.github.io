# Exercise exr-vcp-fluxpot-noscale: the no-scale identity. For K = -3 log(T + Tbar), K_T = -3/(T + Tbar) and
# K_{T Tbar} = 3/(T + Tbar)^2, so K^{T Tbar} K_T K_Tbar = 3 and e^K (K^{T Tbar} |D_T W|^2 - 3 |W|^2) = 0 for a
# T-independent W: the volume modulus is a flat direction of the flux potential.
T, Tb, W = var("T Tb W")
K = -3 * log(T + Tb)
KT, KTb, KTTb = diff(K, T), diff(K, Tb), diff(K, T, Tb)
print("K_T =", KT, "; K_{T Tbar} =", KTTb)
print("K^{T Tbar} K_T K_Tbar =", (KT * KTb / KTTb).simplify_full())
V = exp(K) * ((KT * W) * (KTb * W) / KTTb - 3 * W**2)
print("V for constant W:", V.simplify_full())
