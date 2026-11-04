# Exercise exr-log-values: principal logarithms and principal powers; the three cube roots of -8.
print("Log(1 + i) =", log(1 + I).simplify_full(), " i.e. (1/2) ln 2 + i pi/4:", bool(log(1 + I) == log(2)/2 + I*pi/4))
print("Log(-e) =", log(-e).simplify_full())
print("principal (-1)^(1/3) =", ((-1)^(1/3)).simplify_full(), " numerically", ((-1)^(1/3)).n())
print("principal i^i =", (I^I).simplify_full(), "=", (I^I).n())
z = var("z")
print("all cube roots of -8:", [r for r, m in (z^3 + 8).roots(ring=QQbar)])
print("Sage's principal value (-8)^(1/3):", ((-8)^(1/3)).n(), " (not -2)")
