# Exercise exr-sr-proper-time-limited (Blennow-Ohlsson, Problem 1.38 b): four-velocity and proper time along
# x = 3/2 a t^2, y = 2 a t^2 (c = 1).
var('a t t0', domain='positive')
x = 3/2*a*t^2; y = 2*a*t^2
v2 = (diff(x, t)^2 + diff(y, t)^2).simplify_full()
print("speed^2 =", v2)
g = 1/sqrt(1 - v2)
print("four-velocity =", [g, (g*diff(x, t)).simplify_full(), (g*diff(y, t)).simplify_full(), 0])
F = (5*a*t0*sqrt(1 - 25*a^2*t0^2) + arcsin(5*a*t0))/(10*a)
print("dF/dt0 - sqrt(1 - 25 a^2 t0^2) =", (diff(F, t0) - sqrt(1 - 25*a^2*t0^2)).simplify_full(), "  F(0) =", F.subs(t0=0))
print("F(1/(5a)) =", F.subs(t0=1/(5*a)).simplify_full())
print("numerical check a = 1, t0 = 0.15:", n(F.subs(a=1, t0=0.15)), numerical_integral(sqrt(1 - 25*t^2), 0, 0.15)[0])
