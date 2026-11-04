# Exercise exr-fyd-cutoff-bubble: int_{|l| < Lambda} d^4l/(2 pi)^4 1/(l^2 + Delta)^2 with d^4l = 2 pi^2 l^3 dl, and its
# large-cutoff expansion (1/(16 pi^2))(log(Lambda^2/Delta) - 1) + O(Delta/Lambda^2).
l = var("l"); Lam, D = var("Lambda Delta", domain="positive")
I = 2 * pi ** 2 / (2 * pi) ** 4 * integrate(l ** 3 / (l ** 2 + D) ** 2, l, 0, Lam)
target = (log((Lam ** 2 + D) / D) + D / (Lam ** 2 + D) - 1) / (16 * pi ** 2)
print("integral equals (1/16 pi^2)(log((L^2+D)/D) + D/(L^2+D) - 1):", bool((I - target).simplify_log().simplify_full() == 0))
print("limit of 16 pi^2 I - log(Lambda^2/Delta) as Lambda -> oo:", limit((16 * pi ** 2 * target - log(Lam ** 2 / D)).simplify_log(), Lam, oo))
print("next term: Lambda^2 (16 pi^2 I - log(Lambda^2/Delta) + 1) ->", limit(Lam ** 2 * (16 * pi ** 2 * target - log(Lam ** 2 / D) + 1).simplify_log(), Lam, oo))
