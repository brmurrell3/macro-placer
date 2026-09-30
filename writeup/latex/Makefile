LATEXMK ?= latexmk
PYTHON ?= python3
BUILD := build
MAIN := main
SOURCES := $(MAIN).tex references.bib $(wildcard sections/*.tex) $(wildcard generated/*.tex) $(wildcard figures/*.pdf)

.PHONY: all check clean

all: $(BUILD)/$(MAIN).pdf

check: all
	$(PYTHON) ../../scripts/check_publication_paper.py

$(BUILD)/$(MAIN).pdf: $(SOURCES)
	@mkdir -p $(BUILD)
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -file-line-error \
		-outdir=$(BUILD) $(MAIN).tex

clean:
	$(LATEXMK) -C -outdir=$(BUILD) $(MAIN).tex
