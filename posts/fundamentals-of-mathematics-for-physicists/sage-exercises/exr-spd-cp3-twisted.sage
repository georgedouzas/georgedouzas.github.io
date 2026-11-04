# Exercise exr-spd-cp3-twisted: on CP^3, chi = int c_3, int ch_3(T) and int [A-hat ch(T)]_6, with c(T) = (1 + x)^4,
# p1 = 4 x^2, int x^3 = 1; ch(T) = 4 e^x - 1 since T + O = O(1)^4.
x = var("x")
c = ((1 + x) ^ 4).expand()
chi = c.coefficient(x, 3)
ch = (4 * exp(x) - 1).taylor(x, 0, 3)
ahat = 1 - 4 * x ^ 2 / 24
tw = (ahat * ch).taylor(x, 0, 3).coefficient(x, 3)
print("chi(CP^3) = int c_3 =", chi, "; int ch_3(T) =", ch.coefficient(x, 3), "; int [A-hat ch(T)]_6 =", tw, "; chi/2 =", chi / 2)
