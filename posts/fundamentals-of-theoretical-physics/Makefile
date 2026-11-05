PY ?= python3
SAGE ?= sage

.PHONY: check-text figures lean

check-text:
	$(PY) scripts/check_format.py
	$(PY) scripts/check_definitions.py
	$(PY) scripts/check_figures.py
	$(PY) scripts/check_exercises.py
	$(PY) scripts/check_bib.py

figures:
	@for s in figures/*.sage sage/*.sage; do [ -e "$$s" ] || continue; echo "== $$(basename $$s)"; $(SAGE) "$$s" || exit 1; done

lean:
	cd lean && lake build
