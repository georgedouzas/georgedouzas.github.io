# name: fa-rank
# technique: rank bounds from a non-singular minor (below) and a factorisation A = B * C (above) (matrix_rank recipe),
#   with the matrix read from the goal's !![…] literal
# goal: Matrix.rank !![…] = r for an explicit rational matrix (the Laplacian on homogeneous monomials)
# description: Sage reads the literal, computes the rank, a minor and a rank factorisation; Lean checks the minor's
#   determinant and the product.
import re
text = goal.text
lit = re.search(r'!!\[(.*?)\]', text, re.S).group(1)
rows = [[QQ(sage_eval(x.strip().replace('(', '').replace(')', '').replace(': ℚ', ''))) for x in r.split(',')] for r in lit.split(';')]
a = matrix(QQ, rows)
print('rank:', a.rank())
kind = 'eq'
proof = tactic(rank_proof(a, kind))
