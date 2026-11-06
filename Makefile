# Fast gate: every unit and isasim-level test. `make check_rtl` adds RTL runs.
.PHONY: check check_rtl
check:
	python3 -m pytest -q -m "not rtl and not torch"
check_rtl:
	python3 -m pytest -q -m rtl
