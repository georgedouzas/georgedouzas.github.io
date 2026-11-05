# Fundamentals of Mathematics for Physicists

The tutorial's pages render with the rest of the site. Everything else that belongs to it is here too.

| Path | Contents |
|---|---|
| `*.qmd` | Landing page and chapters |
| `figures/` | Figures, with the Sage script that generates each one |
| `sage-exercises/` | Sage solutions and their recorded output |
| `sage/` | Sage checks for results verified by computation, where there are any |
| `lean/` | Lean package, with certificates and tests in `lean/.veriphy/` |
| `verification/` | Verification manifests and logs |
| `scripts/`, `tests/` | Checkers and their tests |
| `_references/` | Reading notes, tracked, and PDFs, ignored by git |
| `_archive/` | Earlier chapter versions, ignored by git |

## Commands

Run these from this folder:

```sh
make check-text   # formatting, definitions, figures, exercises and bibliography
make figures      # regenerate figures with Sage
make lean         # build the Lean package
```
