# name: la-rank
# technique: rank bounds from a non-singular minor and a rank factorisation (matrix_rank recipe)
# goal: Matrix.rank A = r, r ≤ Matrix.rank A or Matrix.rank A ≤ r, A a literal !![…], a diagonal or the identity
# description: the matrix is read from the goal; Sage computes the rank, a minor and A = B C; Lean checks the minor's
#   determinant and the product.
import re


def _rational(x):
    return QQ(sage_eval(re.sub(r"[()]", "", x).strip()))


def family(text):
    """The explicit family ![![…], …] (or vector ![…]) written in the goal, as lists of rationals."""
    t = re.sub(r"\(([^():]*?) : [^()]*\)", r"\1", text)  # drop type ascriptions
    return [[_rational(x) for x in v.split(",")] for v in re.findall(r"!\[([^\[\]]*)\]", t)]


def literal(text):
    """The first matrix literal !![a, b; c, d] written in the goal, as a list of rows."""
    t = re.sub(r"\(([^():]*?) : [^()]*\)", r"\1", text)
    body = re.search(r"!!\[([^\[\]]*)\]", t).group(1)
    return [[_rational(x) for x in row.split(",")] for row in body.split(";")]


def field(text):
    return "ℝ" if "ℝ" in text else "ℚ"

text = goal.text
if '!![' in text:  # EDIT: read from the goal: a literal, a diagonal ![…] or the identity of Fin n
    MATRIX = literal(text)
elif 'diagonal' in text:
    d = family(text)[0]
    MATRIX = [[d[i] if i == j else 0 for j in range(len(d))] for i in range(len(d))]
else:
    found = re.search(r'Fin (\d+)', text)
    n = int(found.group(1)) if found else int(text.split('=')[-1])  # the identity of the stated rank
    MATRIX = identity_matrix(QQ, n).rows()
a = matrix(QQ, MATRIX)
kind = ('le' if 'rank' in text.split('≤')[0] else 'ge') if '≤' in text else 'eq'
print('rank:', a.rank())
proof = tactic(rank_proof(a, kind))
