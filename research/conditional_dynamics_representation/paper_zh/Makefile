TEXLIVE_BIN := /opt/texlive/2025/bin/x86_64-linux
LATEXMK := $(TEXLIVE_BIN)/latexmk

.PHONY: pdf clean

pdf:
	PATH=$(TEXLIVE_BIN):$$PATH $(LATEXMK) -xelatex -interaction=nonstopmode -halt-on-error main.tex

clean:
	PATH=$(TEXLIVE_BIN):$$PATH $(LATEXMK) -C main.tex
