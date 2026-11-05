"""Audit the Lean library: list its theorems, check that names exist, and check their axioms.

Usage: python3 lean_audit.py        (prints the theorem count and every theorem using a non-standard axiom or
                                     `sorryAx`; exit code 1 if there is one, or if a theorem cannot be resolved)
"""
import glob, os, re, subprocess, sys, tempfile

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
LIB = "KaluzaKlein"
STANDARD = {"propext", "Classical.choice", "Quot.sound"}


def _strip_comments(text):
    """Remove block comments (nested) and line comments, keeping line structure."""
    out, depth, i = [], 0, 0
    while i < len(text):
        if text.startswith("/-", i):
            depth += 1; i += 2; continue
        if depth and text.startswith("-/", i):
            depth -= 1; i += 2; continue
        if depth:
            out.append("\n" if text[i] == "\n" else " "); i += 1; continue
        if text.startswith("--", i):
            j = text.find("\n", i); i = len(text) if j < 0 else j; continue
        out.append(text[i]); i += 1
    return "".join(out)


def sources():
    """Every Lean file of the library, recursively."""
    return sorted(glob.glob(os.path.join(ROOT, LIB, "**", "*.lean"), recursive=True))


def theorems():
    """Fully qualified names of every theorem and lemma in the library."""
    names = []
    for f in sources():
        ns = []
        for line in _strip_comments(open(f).read()).split("\n"):
            m = re.match(r"(namespace|end) (\S+)\s*$", line)
            if m:
                if m.group(1) == "namespace":
                    ns.append(m.group(2))
                elif ns and ns[-1] == m.group(2):
                    ns.pop()
            m = re.match(r"(?:private |protected )?(?:theorem|lemma) (\S+)", line)
            if m:
                names.append(".".join(ns + [m.group(1)]))
    return names


def live_sage_calls():
    """(file, line) of every live `sage_run` tactic call in the library; committed proofs must have none."""
    out = []
    for f in sources():
        for i, line in enumerate(_strip_comments(open(f).read()).split("\n"), 1):
            if re.search(r"\bsage_run\b", line):
                out.append((os.path.relpath(f, ROOT), i))
    return out


def _run(commands):
    """Compile `import KaluzaKlein` followed by the given commands; return Lean's output."""
    with tempfile.NamedTemporaryFile("w", suffix=".lean", delete=False, dir=ROOT, prefix="_audit_") as fh:
        fh.write(f"import {LIB}\n" + "\n".join(commands) + "\n")
        path = fh.name
    env = dict(os.environ, PATH=os.path.expanduser("~/.elan/bin") + ":" + os.environ["PATH"])
    try:
        r = subprocess.run(["lake", "env", "lean", path], cwd=os.path.join(ROOT, "lean"), env=env, capture_output=True, text=True)
    finally:
        os.remove(path)
    return r.stdout + r.stderr


def missing(names):
    """The names among `names` that are not constants of the library."""
    if not names:
        return []
    out = _run([f"#check @{n}" for n in names])
    return [n for n in names if re.search(rf"(Unknown (identifier|constant)|unknown (identifier|constant)) `?{re.escape(n)}`?", out)]


def axioms(names):
    """Map each name to the set of axioms it depends on (None if it could not be resolved)."""
    if not names:
        return {}
    out = _run([f"#print axioms {n}" for n in names])
    found = {n: set(a.replace(" ", "").replace("\n", "").split(",")) - {""} for n, a in
             re.findall(r"'(\S+)' depends on axioms: \[([^\]]*)\]", out)}
    found.update({n: set() for n in re.findall(r"'(\S+)' does not depend on any axioms", out)})
    return {n: found.get(n) for n in names}


if __name__ == "__main__":
    names = theorems()
    ax = axioms(names)
    unresolved = [n for n, a in ax.items() if a is None]
    nonstandard = {n: sorted(a - STANDARD) for n, a in ax.items() if a is not None and a - STANDARD}
    live = live_sage_calls()
    print(f"{len(names)} theorems; unresolved: {unresolved}; non-standard axioms: {nonstandard}; live sage_run: {live}")
    sys.exit(1 if unresolved or nonstandard or live else 0)
