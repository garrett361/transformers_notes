TEX_FILES = $(wildcard *.tex)

all: $(TEX_FILES) bibliography.bib
	xelatex -verbose -file-line-error -shell-escape main.tex
	bibtex main
	xelatex -verbose -file-line-error -shell-escape main.tex
	xelatex -verbose -file-line-error -shell-escape main.tex
	open main.pdf

clean:
	rm *.aux *.dvi *.pdf *.fls *.log *.synctex* *.pyg *.bbl *.brg *.brf *.out *.toc
