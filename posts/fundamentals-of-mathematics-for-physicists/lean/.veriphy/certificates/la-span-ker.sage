# name: la-span-ker
# technique: a span equal to a kernel (independence recipe, KERNEL): generators in the kernel, and a kernel vector's
#   coordinates read off the rows of a non-singular minor
# goal: Submodule.span K (Set.range v) = LinearMap.ker (Matrix.mulVecLin A), v and A written in the goal
# description: Sage checks A v = 0 and that v has the kernel's dimension, and finds the minor; Lean checks the
#   generators by evaluation and the coordinates of a kernel vector by linarith from the kernel equations.
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

left, right = re.split(r'\s=\s', goal.text, maxsplit=1)
VECTORS = family(left)  # EDIT: read from the goal
KERNEL = literal(right)
print('family:', VECTORS, 'kernel of:', KERNEL)
proof = tactic(span_proof(VECTORS, kernel_of=matrix(QQ, KERNEL), field=field(goal.text)))
