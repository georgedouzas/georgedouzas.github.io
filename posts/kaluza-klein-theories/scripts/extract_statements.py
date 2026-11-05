"""Draft chapters under tutorial/_drafts/ (material not yet split into the numbered chapters) count as chapters here.
Extract the Lean statements the tutorial shows into tutorial/_generated/statements.json (Quarto pre-render hook).

A chapter shows a Lean statement with the shortcode `{{< lean <FullyQualifiedName> >}}` (tutorial/lean.lua). This
script scans every chapter for the shortcode, finds each declaration in the Lean library (namespace-aware), and
writes its docstring and statement (the text up to `:=`, proofs omitted) with the source location. A name that no
declaration matches is an error: the script exits 1, naming it, and Quarto aborts the render. The shortcode also
fails the render if it is asked for a name the file lacks.

Usage: python3 extract_statements.py        (from anywhere; paths are resolved from this file)
"""
import glob, json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import lean_audit

ROOT = os.path.normpath(os.path.join(HERE, ".."))
TUTORIAL = ROOT
OUT = os.path.join(TUTORIAL, "_generated", "statements.json")
SHORTCODE = re.compile(r"\{\{<\s*lean\s+([\w.']+)\s*>\}\}")
DECL = re.compile(r"^(?:@\[[^\]]*\]\s*)?(?:private |protected |noncomputable )*(theorem|lemma|def|abbrev|structure|inductive) (\S+)")


def declarations():
    """Map each fully qualified name to {file, namespace, doc, statement}."""
    out = {}
    for f in lean_audit.sources():
        lines = open(f).read().split("\n")
        ns, doc, i = [], [], 0
        while i < len(lines):
            line = lines[i]
            m = re.match(r"(namespace|end) (\S+)\s*$", line)
            if m:
                if m.group(1) == "namespace":
                    ns.append(m.group(2))
                elif ns and ns[-1] == m.group(2):
                    ns.pop()
                doc = []; i += 1; continue
            if line.startswith("/--"):
                doc = [line]
                while not lines[i].rstrip().endswith("-/"):
                    i += 1; doc.append(lines[i])
                i += 1; continue
            m = DECL.match(line)
            if m:
                stmt = [line]
                while ":=" not in stmt[-1] and i + 1 < len(lines) and lines[i + 1].strip() and not DECL.match(lines[i + 1]):
                    i += 1; stmt.append(lines[i])
                text = "\n".join(stmt)
                text = text[:text.index(":=")].rstrip() if ":=" in text else text
                name = ".".join(ns + [m.group(2)])
                out[name] = {"file": os.path.relpath(f, ROOT), "namespace": ".".join(ns), "doc": "\n".join(doc),
                             "statement": text}
                doc = []
            elif line.strip():
                doc = []
            i += 1
    return out


def referenced():
    names = set()
    for f in (glob.glob(os.path.join(TUTORIAL, "*.qmd")) + glob.glob(os.path.join(TUTORIAL, "_drafts", "*.qmd"))):
        names.update(SHORTCODE.findall(open(f).read()))
    return sorted(names)


def main():
    """Strict unless KK_DRAFT=1: a missing declaration fails the render (`make render` sets KK_STRICT=1 and never
    drafts); in draft mode (a plain `quarto render` while writing) it becomes a visible "pending" box instead."""
    draft = os.environ.get("KK_DRAFT") == "1" and os.environ.get("KK_STRICT") != "1"
    decls, names = declarations(), referenced()
    missing = [n for n in names if n not in decls]
    if missing and not draft:
        for n in missing:
            print(f"extract_statements: no Lean declaration named {n}")
        print("extract_statements: set KK_DRAFT=1 to render a draft with pending boxes instead")
        sys.stdout.flush()
        return 1
    out = {n: decls[n] for n in names if n in decls}
    for n in missing:
        print(f"extract_statements: DRAFT, no Lean declaration named {n} (pending box)")
        out[n] = {"file": "", "namespace": "", "doc": "", "statement": "", "pending": True}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1, sort_keys=True)
    print(f"extract_statements: wrote {len(names)} statement(s) to tutorial/_generated/statements.json" + (f" ({len(missing)} pending)" if missing else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
