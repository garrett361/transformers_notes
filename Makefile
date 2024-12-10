TEX_FILES = $(wildcard *.tex)

all: $(TEX_FILES) bibliography.bib
	xelatex -file-line-error -shell-escape main.tex
	bibtex main
	xelatex -file-line-error -shell-escape main.tex
	xelatex -file-line-error -shell-escape main.tex
	open main.pdf

clean:
	rm *.aux *.dvi *.pdf *.fls *.log *.synctex* *.pyg *.bbl *.brg *.brf *.out *.toc
