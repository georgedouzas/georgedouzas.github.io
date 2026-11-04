# Exercise exr-sr-rescue (Blennow-Ohlsson, Problem 1.46): a light signal to an approaching station and a rescue ship sent
# back, in the frame of the sender (days and light-days, c = 1).
var('t')
t1 = solve(t == 1 - t/4, t)[0].rhs()
w = -(3/4 + 1/4)/(1 + (3/4)*(1/4))
t2 = solve(t1 + w*(t - t1) == 0, t)[0].rhs()
print("signal arrives at t =", t1, " rescue velocity =", w, " rescue arrives at t =", t2, "=", n(t2), "days")
