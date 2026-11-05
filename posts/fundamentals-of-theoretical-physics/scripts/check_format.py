"""Check the chapter format of design D19-D21 on the given chapter files (default: all tutorial/part-*.qmd).

A chapter file is a thematic group whose `## ` sections are the single-topic units. Rules:
  - at most six definition blocks per section, each with a one-notion title;
  - after every definition, theorem, proposition, lemma or corollary block, before the next such block or the next
    section heading: at least ten `exm-` blocks and ten `exr-` blocks in a raised chapter or for a new statement,
    three otherwise (scripts/density.py), each exercise followed by a collapsed solution callout;
  - at least one figure (`![...](figures/...)`) per section that has a definition;
  - the file ends with a section whose heading starts with "Exercises" containing at least three `exr-` blocks
    with solutions (the integrative exercises);
  - in Parts I and II the chapter introduction (text before the first `## `) does not mention the main theme;
  - Lean statements only inside collapsed Verification callouts (D29);
  - connective prose (D25): consecutive blocks of a section are separated by at least two sentences (twenty words)
    of running text; an exercise's solution callout counts as part of the exercise.

Usage: python3 check_format.py [files...] [--backlog]     exit code 1 on any problem; --backlog prints, per chapter,
       the examples and exercises missing for ten per statement
"""
import glob, os, re, sys
import fold_blocks  # change kk-csdr-presentation: statement boxes and folds
import density  # change kk-csdr-ten-examples: ten examples and exercises per statement

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
TUTORIAL = ROOT
BLOCK = re.compile(r"^:::+ \{#((?:def|exm|prp|thm|lem|cor|exr)-[A-Za-z0-9-]+)[^}]*\}\n(?:#### (.*)\n)?", re.M)
SECTION = re.compile(r"^## (.*)$", re.M)
FIG = re.compile(r"!\[.*?\]\(figures/[\w./-]+\)", re.S)
# the three levels a Veriphy part may name (design D29)
LEVEL_FORM = re.compile(r"[Rr]ecipe `[^`]+`, certificate `[^`]+`|Test \((?:kernel-checked|oracle), \d+ instances\)")
VERIFICATION = re.compile(r"^:::+ \{\.callout-tip[^}]*\}\n.*?^:::+[ \t]*$", re.M | re.S)
SOLUTION = re.compile(r"^\s*:::+ \{\.callout-note[^}]*collapse=\"true\"[^}]*\}", re.M)
STATEMENT = ("def-", "thm-", "prp-", "lem-", "cor-")
BACKLOG = []  # (file, statement, examples missing for ten, exercises missing for ten), for --backlog
THEME = re.compile(r"Kaluza|CSDR|coset space dimensional reduction|Part (III|IV|V|VI)\b", re.I)


def wrapper_problems(name, raw):
    """Change kk-csdr-presentation: every statement is in a statement box, every example and exercise in a fold."""
    found, stack = [], []
    for i, l in enumerate(raw.split("\n"), 1):
        w = fold_blocks.WRAP_OPEN.match(l)
        if w:
            stack.append(w.group(1))
            continue
        m = re.match(r"^:::+\s*\{#(def|thm|prp|lem|cor|exm|exr)-([\w-]+)", l)
        if m:
            need = "statement-box" if m.group(1) in ("def", "thm", "prp", "lem", "cor") else (
                "examples-fold" if m.group(1) == "exm" else "exercises-fold")
            if need not in stack:
                found.append(("blocks are boxed and folded", f"{name}:{i}: {m.group(1)}-{m.group(2)} is not inside a {need} (run scripts/fold_blocks.py)"))
        if fold_blocks.OPEN.match(l):
            stack.append(None)
        elif fold_blocks.CLOSE.match(l) and stack:
            stack.pop()
    return found


def check_file(f):
    out = []
    name = os.path.basename(f)
    raw = open(f, encoding="utf-8").read()
    text = "\n".join(fold_blocks.unfold(raw.split("\n")))
    out += wrapper_problems(name, raw)
    if "To be written" in text:
        # a scaffolded `##` part (D24: headings and scopes before the content) is not checked yet; the rest of the
        # chapter is. A chapter whose parts are all scaffolds is skipped.
        parts = re.split(r"(?m)^(?=## )", text)
        kept = [p for p in parts if "To be written" not in p]
        if len(kept) <= 1 or not any(p.startswith("## ") for p in kept):
            return out
        text = "".join(kept)
    secs = [(m.group(1).strip(), m.start()) for m in SECTION.finditer(text)]
    bounds = [(t, s, (secs[i + 1][1] if i + 1 < len(secs) else len(text))) for i, (t, s) in enumerate(secs)]
    blocks = [(m.group(1), (m.group(2) or "").strip(), m.start(), m.end()) for m in BLOCK.finditer(text)]
    # introduction
    if name.startswith(("part-i-", "part-ii-")) and secs:
        intro = text[:secs[0][1]]
        intro = intro.split("\n", 1)[1] if "\n" in intro else intro
        if THEME.search(intro):
            out.append(("introduction stands on its own", f"{name}: the introduction mentions the main theme"))
    # per section
    for title, s, e in bounds:
        inner = [b for b in blocks if s < b[2] < e]
        # a main-theme chapter (design D27) nests former chapters as `## ` parts whose sections are `### `: the
        # single-topic unit is then the `### ` section
        subs = [(m.group(1).strip(), s + m.start()) for m in re.finditer(r"^### (.*)$", text[s:e], re.M)]
        units = [(t2, st, (subs[i + 1][1] if i + 1 < len(subs) else e)) for i, (t2, st) in enumerate(subs)] \
            if re.search(r"^### Exercises", text[s:e], re.M) else [(title, s, e)]
        for ut, us, ue in units:
            udefs = [b for b in inner if b[0].startswith("def-") and us < b[2] < ue]
            if len(udefs) > 6:
                out.append(("at most six definitions per section", f"{name} / {ut}: {len(udefs)} definitions"))
            if udefs and not FIG.search(text[us:ue]):
                out.append(("figure per section", f"{name} / {ut}: no figure"))
        stmts = [b for b in inner if b[0].startswith(STATEMENT)]
        for k, (label, ttl, pos, end) in enumerate(stmts):
            if ("," in ttl) or (" and " in ttl.lower()) or (" or " in ttl.lower()):
                if label.startswith("def-"):
                    out.append(("one notion per definition", f"{name}: {label} ('{ttl}') names several notions"))
            nxt = stmts[k + 1][2] if k + 1 < len(stmts) else e
            exm = [b for b in inner if b[0].startswith("exm-") and pos < b[2] < nxt]
            exr = [b for b in inner if b[0].startswith("exr-") and pos < b[2] < nxt]
            need = density.minimum(name, label)
            if len(exm) < need:
                out.append(("examples per statement", f"{name}: {label} has {len(exm)} example(s), needs {need}"))
            if len(exr) < need:
                out.append(("exercises per statement", f"{name}: {label} has {len(exr)} exercise(s), needs {need}"))
            BACKLOG.append((name, label, max(0, density.TEN - len(exm)), max(0, density.TEN - len(exr))))
    # connective prose (D25): at least two sentences and twenty words between consecutive blocks of a section
    CLOSE = re.compile(r"^:::+[ \t]*$", re.M)
    def block_close(start):
        m = CLOSE.search(text, start)
        return m.end() if m else len(text)
    def after_solution(end):
        rest = text[end:]
        m = SOLUTION.match(rest.lstrip("\n"))
        if not m:
            return end
        sol_start = end + (len(rest) - len(rest.lstrip("\n")))
        return block_close(sol_start + m.end())
    for title, s, e in bounds:
        inner = [b for b in blocks if s < b[2] < e]
        prev_end = text.find("\n", s) + 1          # the first block is introduced by the section's own prose
        for label, ttl, pos, end in inner:
            if prev_end is not None:
                gap = text[prev_end:pos]
                gap = VERIFICATION.sub(" ", gap)                                     # verification callouts
                gap = re.sub(r"^#+ .*$", " ", gap, flags=re.M)             # headings
                gap = re.sub(r"!\[.*?\]\(.*?\)(\{[^}]*\})?", " ", gap, flags=re.S)  # figures
                gap = re.sub(r"\{\{<.*?>\}\}", " ", gap)                     # shortcodes
                words = len(gap.split())
                sentences = len(re.findall(r"[.!?](\s|$)", gap))
                if words < 20 or sentences < 2:
                    out.append(("connective prose between blocks", f"{name}: {label} follows the previous block with {words} word(s) of prose (need two sentences, twenty words)"))
            close = block_close(end)
            prev_end = after_solution(close) if label.startswith("exr-") else close
    # design D29: Lean statements are shown only inside a collapsed Verification callout
    stripped = VERIFICATION.sub("", text)
    for m in re.finditer(r"\{\{<\s*lean\s+([\w.']+)", stripped):
        out.append(("Lean only in verification callouts", f"{name}: lean shortcode {m.group(1)} outside a Verification callout"))
    # design D29: every Verification callout is a Veriphy part that decomposes into a Lean part and a Sage part, in
    # that order: "**Veriphy.**" (recipe and certificate), "**Lean.**" (the `lean` shortcodes), "**Sage.**" (the
    # oracle computation). A part may read "Pending" until the claim is certified; `--strict` (the gate) forbids it.
    for m in VERIFICATION.finditer(text):
        body = m.group(0)
        line = text[:m.start()].count("\n") + 1
        pos = [body.find(f"**{p}.**") for p in ("Veriphy", "Lean", "Sage")]
        if min(pos) < 0 or pos != sorted(pos):
            out.append(("Verification = Veriphy, Lean, Sage", f"{name}:{line}: Verification callout lacks the parts **Veriphy.**, **Lean.**, **Sage.** in that order"))
        elif "--strict" in sys.argv and re.search(r"\bPending\b", body):
            out.append(("Verification = Veriphy, Lean, Sage", f"{name}:{line}: Verification callout has a pending part"))
        elif "--strict" in sys.argv and not LEVEL_FORM.search(body[pos[0]:pos[1]]):
            out.append(("Verification = Veriphy, Lean, Sage", f"{name}:{line}: the Veriphy part names no level (Recipe `…`, certificate `…` / Test (kernel-checked, n instances) / Test (oracle, n instances))"))
        elif re.search(r"\{\{<\s*lean", body[:pos[1]]) or re.search(r"\{\{<\s*lean", body[pos[2]:]):
            out.append(("Verification = Veriphy, Lean, Sage", f"{name}:{line}: lean shortcode outside the **Lean.** part"))
    # a Lean name with a prime breaks Quarto's shortcode parser (the whole book fails to render)
    for m in re.finditer(r"\{\{<\s*lean\s+([\w.]*'[\w.']*)", text):
        out.append(("Lean shortcode names without primes", f"{name}: lean shortcode {m.group(1)} has a prime; rename the Lean declaration"))
    # design D29: the main text never refers to verification; that material lives only in the Verification callouts.
    # Solutions may show Sage computations (teaching material) but not Lean or Veriphy; figure captions may name their
    # generating script.
    main = VERIFICATION.sub("", text)
    main = re.sub(r"!\[.*?\]\(.*?\)(\{[^}]*\})?", " ", main, flags=re.S)
    sols = [m for m in re.finditer(r"^:::+ \{\.callout-note[^}]*\}\n.*?^:::+[ \t]*$", main, re.M | re.S)]
    def scan(chunk, words, where):
        for m in words.finditer(chunk):
            line = chunk[:m.start()].count("\n")
            out.append(("main text does not refer to verification", f"{name}: '{m.group(0)}' in {where}: {chunk.splitlines()[line][:90] if chunk.splitlines() else ''}"))
    hard = re.compile(r"Verified in Lean|\bLean\b|Veriphy|KaluzaKlein\.|\bcertificates?\b|\brecipes?\b|\bregistry\b|\bmathlib\b|kernel-checked|[Cc]hecked by exact computation|[Vv]erified (?:results|values)|figure script")
    soft = re.compile(r"(?:Sage|SageManifolds) confirms|[Cc]hecked (?:in|by|with) Sage|[Vv]erified (?:in|by|with) Sage|[Cc]onfirmed (?:in|by|with) Sage|`[\w-]+\.sage`")
    outside = main
    for m in reversed(sols):
        outside = outside[:m.start()] + outside[m.end():]
    scan(outside, hard, "main text")
    scan(outside, soft, "main text")
    for m in sols:
        scan(m.group(0), hard, "a solution")
    # a block title must not contain "]": it ends the LaTeX optional argument of the PDF environment (use \rbrack)
    for m in re.finditer(r"^:::+ \{#((?:def|thm|prp|lem|cor|exm|exr)-[\w-]+)[^}]*\}\n#+ (.*\].*)$", text, re.M):
        out.append(("block titles build in the PDF", f"{name}: {m.group(1)} has ']' in its title: {m.group(2)[:60]}"))
    # an inline formula has no space just inside its dollars, and must not break so that the next line starts with a list or quote marker ("+ x", "- x", "> x",
    # "2. x"): Pandoc reads it as a new item and the formula is cut in two (the PDF build fails)
    lines, inm, disp, fence = text.split("\n"), False, False, False
    for i, l in enumerate(lines):
        s = l.strip()
        if s.startswith("```"):
            fence, inm = not fence, False
            continue
        if fence or not s or s.startswith(":::"):
            inm = False
            continue
        j = 0
        while j < len(l):
            if l[j] == "\\":
                j += 2
                continue
            if l[j:j + 2] == "$$":
                disp, j = not disp, j + 2
                continue
            if l[j] == "$" and not disp:
                # Pandoc reads "$ x$" and "$x $" as dollar signs, not as a formula
                if (not inm and (j + 1 == len(l) or l[j + 1] in " \t")) or (inm and (j == 0 or l[j - 1] in " \t")):
                    out.append(("formulas have no space inside their dollars",
                                f"{name}: line {i + 1}: '{l[max(0, j - 25):j + 15].strip()}'"))
                inm = not inm
            j += 1
        if inm and not disp and i + 1 < len(lines) and re.match(r"\s*(\d+\.|[+\-*>])(\s|$)", lines[i + 1]):
            out.append(("formulas do not break before a list marker",
                        f"{name}: line {i + 2} starts with '{lines[i + 1].strip()[:12]}' inside an inline formula"))
    # a cross-reference must not be split across lines at a hyphen
    for m in re.finditer(r"@[a-z]+-[\w-]*-\n", text):
        out.append(("cross-reference on one line", f"{name}: {m.group(0).strip()} is split across lines"))
    # every exercise has a solution
    for label, ttl, pos, end in blocks:
        if label.startswith("exr-"):
            after = text[end:]
            close = re.search(r"^:::+[ \t]*\n", after, re.M)
            rest = after[close.end():] if close else after
            if not SOLUTION.match(rest.lstrip("\n")):
                out.append(("exercise has a folded solution", f"{name}: {label} is not followed by a collapsed solution callout"))
    # integrative exercises at the end (an optional 'Summary of verified results' section may follow)
    content = [b for b in bounds if not b[0].lower().startswith("summary")]
    last_ok = bool(content) and content[-1][0].lower().startswith("exercises")
    if content and not last_ok:
        # a main-theme chapter may end with the `### Exercises` of its last `## ` part
        t, s0, e0 = content[-1]
        subs = [(m.group(1).strip(), s0 + m.start()) for m in re.finditer(r"^### (.*)$", text[s0:e0], re.M)]
        subs = [x for x in subs if not x[0].lower().startswith("summary")]
        if subs and subs[-1][0].lower().startswith("exercises"):
            nxt = [m.start() + subs[-1][1] for m in re.finditer(r"^### ", text[subs[-1][1] + 4:e0], re.M)]
            content = content[:-1] + [(subs[-1][0], subs[-1][1], e0)]
            last_ok = True
    if not last_ok:
        out.append(("chapter ends with exercises", f"{name}: last section is not 'Exercises'"))
    else:
        t, s, e = content[-1]
        n = sum(1 for b in blocks if b[0].startswith("exr-") and s < b[2] < e)
        if n < 3:
            out.append(("three integrative exercises", f"{name}: {n} exercise(s) in the final section"))
    return out


if __name__ == "__main__":
    files = [a for a in sys.argv[1:] if not a.startswith("-")] or sorted(glob.glob(os.path.join(TUTORIAL, "part-*.qmd")))
    found = []
    for f in files:
        found += check_file(f)
    # identifiers must be unique across the whole book (Quarto resolves cross-references book-wide)
    seen = {}
    for f in sorted(glob.glob(os.path.join(TUTORIAL, "*.qmd"))):
        for m in re.finditer(r"\{#((?:def|exm|exr|prp|thm|lem|cor|sec|fig|eq)-[\w-]+)", open(f, encoding="utf-8").read()):
            seen.setdefault(m.group(1), []).append(os.path.basename(f))
    for ident, where in sorted(seen.items()):
        if len(where) > 1 and any(os.path.basename(x) in where for x in files):
            found.append(("identifiers are unique", f"{ident} is defined in {', '.join(where)}"))
    if "--backlog" in sys.argv:
        per = {}
        for name, label, mx, mr in BACKLOG:
            if mx or mr:
                print(f"{name}: {label} lacks {mx} example(s) and {mr} exercise(s) for ten")
            a = per.setdefault(name, [0, 0, 0])
            a[0] += 1; a[1] += mx; a[2] += mr
        for name, (n, mx, mr) in sorted(per.items()):
            print(f"{name}: {n} statements, {mx} examples and {mr} exercises missing for ten")
    for n, msg in found:
        print(f"[{n}] {msg}")
    print(f"{len(files)} chapter file(s); problems {len(found)}")
    sys.exit(1 if found else 0)
