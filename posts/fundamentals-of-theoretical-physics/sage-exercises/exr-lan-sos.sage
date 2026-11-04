# Exercise exr-lan-sos: f(m) = a (T - T_c) m^2 + b m^4. Below T_c write f - f_min as b (m^2 - m_0^2)^2; compute the
# entropy and heat capacity on both sides and the jump of C at T_c.
m, T, Tc, a, b = var("m T T_c a b"); assume(a > 0, b > 0, T > 0, Tc > 0)
f = a * (T - Tc) * m ** 2 + b * m ** 4
m0sq = a * (Tc - T) / (2 * b)
fmin = f.subs(m=sqrt(m0sq)).simplify_full()
print("f(m_0) =", fmin.factor(), "; f - f(m_0) - b (m^2 - m_0^2)^2 =", (f - fmin - b * (m ** 2 - m0sq) ** 2).expand())
S_below = -diff(fmin, T); C_below = T * diff(S_below, T)
print("below T_c: S - S_0 =", S_below.factor(), " C - C_0 =", C_below, "; above T_c: S - S_0 = 0, C - C_0 = 0")
print("entropy continuous at T_c:", bool(S_below.subs(T=Tc) == 0), "; jump Delta C =", C_below.subs(T=Tc))
