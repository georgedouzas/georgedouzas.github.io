"""Author Lean certificates with Veriphy's `sage_run` and commit only their Sage-free replacements.

A Lean file under KaluzaKlein/ marks each proof that needs Sage with `VERIPHY[<script>]` in tactic position, where
`.veriphy/certificates/<script>.sage` is the certificate script, and each claim to be tested (Veriphy's `sage_test`
fallback, when no certificate is feasible) with a line `VERIPHY_TEST[<claim>, <script>]` at command level after the
claim. A test's replacement is the kernel-checked instances and the `veriphy_tested` attribute; its record is written
to `.veriphy/tests/<claim>.json`. Committed files that carry a replacement import `Veriphy.Kit.Levels` (Sage-free).
Lean runs go through the shared lock of scripts/lean_lock.py. `python3 .veriphy/certify.py <file.lean> ...`:

1. writes an authoring copy `.veriphy/_authoring_<name>.lean` with `import Veriphy` and each marker replaced by
   `sage_run file "<script>.sage"`;
2. compiles it from the Lake workspace root with `lake env lean`, where Sage finds the certificate and the Lean
   kernel checks it;
3. replaces each marker in the original file by the `Sage-free replacement:` that `sage_run` printed, followed by a
   comment `-- certificate: <script>.sage` so the proof can be regenerated.

The committed file therefore builds with Lean and mathlib alone. Needs SageMath with `veriphy` installed and elan.
With `--check`, the replacements are compared with the committed proofs instead of written (exit 1 if they differ).
"""
import json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))  # .veriphy/
ROOT = os.path.normpath(os.path.join(HERE, ".."))  # the Lake workspace root
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import lean_lock
SCRIPTS = os.path.join(HERE, "certificates")
MARK = re.compile(r"VERIPHY\[([\w-]+)\]|^VERIPHY_TEST\[([\w.']+),\s*([\w-]+)\][ \t]*$", re.M)
LEVELS_IMPORT = "import Veriphy.Kit.Levels"
REPLACEMENT = re.compile(r"Sage-free replacement: \n  \[apply\] (.*?)(?=\n\S|\Z)", re.S)
POSITIONED = re.compile(r"^\S*?_authoring_[\w-]+\.lean:(\d+):(\d+): (?:info: )?Sage-free replacement: \n  \[apply\] (.*?)(?=\n\S|\Z)", re.S | re.M)
COMMITTED = re.compile(r"-- certificate: ([\w-]+)\.sage")


def _reindent(rep, column):
    """Re-indent a replacement printed by `sage_run` to its marker: continuation lines two columns deeper."""
    lines = rep.rstrip().split("\n")
    if lines[0].startswith("(") and lines[-1].endswith(")"):  # a parenthesised block: drop the parentheses
        lines[0] = lines[0][1:]
        lines[-1] = lines[-1][:-1]
        if not lines[0].strip():
            lines = lines[1:]
            cut = min((len(l) - len(l.lstrip()) for l in lines if l.strip()), default=0)
            return "\n".join([lines[0][cut:]] + [" " * column + l[cut:] if l.strip() else "" for l in lines[1:]])
    rest = [l for l in lines[1:] if l.strip()]
    cut = min((len(l) - len(l.lstrip()) for l in rest), default=0)
    return "\n".join([lines[0]] + [" " * (column + 2) + l[cut:] if l.strip() else "" for l in lines[1:]])


def _commands(rep):
    """A test's replacement (commands printed with every line after the first indented) at column 0."""
    lines = [l.rstrip() for l in rep.rstrip().split("\n")]
    heads = [l for l in lines[1:] if re.match(r"\s*(example|attribute)\b", l)]
    cut = min((len(l) - len(l.lstrip()) for l in heads), default=0)
    return "\n".join([lines[0]] + [l[cut:] if l[:cut].strip() == "" else l for l in lines[1:]])


def _offset(text, line):
    """Character offset of the start of 1-based `line` in `text`."""
    pos = 0
    for _ in range(line - 1):
        pos = text.find("\n", pos) + 1
        if pos == 0:
            return len(text)
    return pos


def _script(m):
    """The certificate or test script named by a marker."""
    return m.group(1) or m.group(3)


def compile_authoring(text, name):
    """Compile the authoring copy of `text`; return the list of replacements in marker order."""
    sites = list(MARK.finditer(text))
    for m in sites:
        script = os.path.join(SCRIPTS, _script(m) + ".sage")
        assert os.path.exists(script), f"missing certificate script {script}"
    imports = [m.end() for m in re.finditer(r"^import .*$", text, re.M)]
    cut = imports[-1] if imports else 0
    authoring = text[:cut] + "\nimport Veriphy" + text[cut:]
    authoring = MARK.sub(lambda m: f'sage_run file "{m.group(1)}.sage"' if m.group(1) else
                         f'sage_test {m.group(2)} file "{m.group(3)}.sage"', authoring)
    scratch = os.path.join(HERE, f"_authoring_{name}.lean")
    with open(scratch, "w") as f:
        f.write(authoring)
    env = dict(os.environ, PATH=os.path.expanduser("~/.elan/bin") + ":" + os.environ["PATH"])
    # `--json` gives every message with its position: the `Sage-free replacement` infos are keyed by line, so the
    # certificates that succeeded are kept even when another marker of the file fails
    r = lean_lock.run(["lake", "env", "lean", "--json", os.path.relpath(scratch, ROOT)], cwd=ROOT, env=env,
                      capture_output=True, text=True)
    msgs, other = [], []
    for line in (r.stdout + r.stderr).split("\n"):
        try:
            msgs.append(json.loads(line))
        except ValueError:
            if line.strip():
                other.append(line)
    log = "\n".join([f"{m.get('fileName', '')}:{m['pos']['line']}:{m['pos']['column']}: {m['severity']}: {m['data']}"
                     for m in msgs if 'pos' in m] + other)
    by_line = {}
    for m in msgs:
        if m.get("severity") == "information" and "Sage-free replacement:" in m.get("data", ""):
            hit = REPLACEMENT.search(m["data"])
            if hit:
                by_line[m["pos"]["line"]] = hit.group(1)
    marker_lines = [text[:m.start()].count("\n") + 1 for m in MARK.finditer(text)]
    # the authoring copy has the same markers, one line lower (the inserted `import Veriphy`)
    partial = [by_line.get(l + 1) for l in marker_lines]
    # `sage_run` prints its replacement before Lean has checked it against the goal: reject a replacement when an
    # error lies in its declaration (between the previous and the next marker, in authoring-copy lines)
    errs = [m["pos"]["line"] for m in msgs if m.get("severity") == "error" and "pos" in m]
    bounds = [0] + [l + 1 for l in marker_lines] + [10 ** 9]
    for k in range(len(partial)):
        lo, hi = bounds[k], bounds[k + 2]
        decl = authoring.rfind("\ntheorem ", 0, _offset(authoring, bounds[k + 1]))
        lo = max(lo, authoring[:decl].count("\n")) if decl >= 0 else lo
        if any(lo < e < hi for e in errs):
            partial[k] = None
    if r.returncode != 0 or ": error" in log or any(p is None for p in partial):
        lines = log.split("\n")
        hits = [i for i, l in enumerate(lines) if ": error" in l or "error:" in l]
        shown = set()
        for i in hits[:12]:
            for j in range(i, min(i + 14, len(lines))):
                if j not in shown:
                    shown.add(j); print(lines[j][:400])
            print("...")
        print(f"[{len(hits)} error line(s); {sum(p is not None for p in partial)} of {len(sites)} replacement(s) succeeded]")
        os.remove(scratch)
        return partial
    assert all(p is not None for p in partial), (partial, log[:3000])
    os.remove(scratch)
    return partial


def certify(path, check=False):
    """Replace the markers of the Lean file at `path` by Sage-free proofs (or compare, with `check`)."""
    text = open(path).read()
    name = os.path.splitext(os.path.relpath(path, ROOT))[0].replace(os.sep, "_").replace(".", "_")
    if check:
        # rebuild the marker form from the committed file: each committed proof is the text between the tactic
        # position and its `-- certificate:` comment; re-author and compare
        marked = re.sub(r"(?s)(:= by\n)(.*?)\s*-- certificate: ([\w-]+)\.sage", lambda m: f"{m.group(1)}  VERIPHY[{m.group(3)}]", text)
        reps = compile_authoring(marked, name)
        committed = re.findall(r"(?s):= by\n(.*?)\s*-- certificate: [\w-]+\.sage", text)
        norm = lambda s: " ".join(s.split())
        bad = [i for i, (c, r) in enumerate(zip(committed, reps)) if norm(c) != norm(_reindent(r, 2))]
        print(f"{path}: {len(reps)} certificate(s), {len(bad)} differ")
        return not bad
    reps = compile_authoring(text, name)
    it = iter(reps)

    def sub(m):
        column = m.start() - (text.rfind("\n", 0, m.start()) + 1)
        rep = next(it)
        if rep is None:  # this marker's certificate failed: keep the marker for the next run
            return m.group(0)
        if m.group(2):  # a test: the replacement is a sequence of commands, moved to column 0
            return _commands(rep) + f"\n-- test: {m.group(3)}.sage"
        return _reindent(rep, column) + f"\n{' ' * column}-- certificate: {m.group(1)}.sage"

    final = MARK.sub(sub, text)
    if any(r is not None for r in reps) and LEVELS_IMPORT not in final:
        imports = [m.end() for m in re.finditer(r"^import .*$", final, re.M)]
        cut = imports[-1] if imports else 0
        final = final[:cut] + ("\n" if cut else "") + LEVELS_IMPORT + ("" if cut else "\n") + final[cut:]
    with open(path, "w") as f:
        f.write(final)
    done = sum(r is not None for r in reps)
    print(f"wrote {os.path.relpath(path, ROOT)} with {done} Veriphy replacement(s)" + ("" if done == len(reps) else f"; {len(reps) - done} marker(s) left"))
    return done == len(reps)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    check = "--check" in sys.argv
    ok = all([certify(os.path.abspath(p), check) for p in args])
    sys.exit(0 if ok else 1)
