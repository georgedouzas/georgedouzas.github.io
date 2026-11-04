# Exercise exr-homeomorphism-tan: tan/arctan are mutually inverse, and x/(1 - x^2) is increasing on (-1, 1).
x = var("x")
print("tan(arctan(x)) =", tan(arctan(x)).simplify_full())
g = x / (1 - x^2)
print("g' =", diff(g, x).simplify_full())
print("[PASS] tan(arctan x) = x, and g' = (1 + x^2)/(1 - x^2)^2 > 0 on (-1, 1) with g -> +-infinity at +-1:",
      bool(tan(arctan(x)).simplify_full() == x) and bool((diff(g, x) - (1 + x^2) / (1 - x^2)^2).simplify_full() == 0) and bool(limit(g, x=1, dir="-") == +Infinity))
