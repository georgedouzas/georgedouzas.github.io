# Exercise exr-fan-norm-axioms: N(x) = (sqrt|x1| + sqrt|x2|)^2 on R^2 is homogeneous but violates the triangle
# inequality; the 1- and sup-norms satisfy it on random vectors; the unit "ball" of N is not convex.
N = lambda x: (sqrt(abs(x[0])) + sqrt(abs(x[1])))^2
print("N(1,0) =", N((1, 0)), " N(0,1) =", N((0, 1)), " N(1,1) =", N((1, 1)), " > N(1,0) + N(0,1):", N((1, 1)) > 2)
print("homogeneity N(-3 x) = 3 N(x) at x = (2, 5):", bool(N((-6, -15)) == 3 * N((2, 5))))
set_random_seed(1)
vs = [vector(CC, [CC.random_element() for _ in range(4)]) for _ in range(200)]
print("triangle inequality for the 1-norm and the sup-norm on 100 random pairs in C^4:",
      all(((u + v).norm(1) <= u.norm(1) + v.norm(1) + 1e-12) and ((u + v).norm(Infinity) <= u.norm(Infinity) + v.norm(Infinity) + 1e-12) for u, v in zip(vs[::2], vs[1::2])))
print("midpoint of (1,0) and (0,1) has N =", N((1 / 2, 1 / 2)), "> 1: the set N <= 1 is not convex")
