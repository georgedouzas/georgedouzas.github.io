# name: la-dim-span
# technique: finrank of the span of an explicit family (dimension recipe, KIND = 'span')
# goal: Module.finrank K ↥(Submodule.span K (Set.range ![![…], …])) = r, the family written in the goal
# description: the family is read from the goal; Sage computes its rank and an echelon basis; Lean rewrites the span
#   to the echelon basis (coordinates) and uses independence from a non-singular minor.
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

VECTORS = family(goal.text)  # EDIT: read from the goal
print('family:', VECTORS, 'rank:', matrix(QQ, VECTORS).rank())
proof = tactic(span_dimension_proof(VECTORS, field(goal.text)))
