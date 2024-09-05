TEX_FILES = $(wildcard *.tex)

all: $(TEX_FILES) bibliography.bib
	xelatex -verbose -file-line-error -shell-escape decoder_only.tex
	bibtex decoder_only
	xelatex -verbose -file-line-error -shell-escape decoder_only.tex
	xelatex -verbose -file-line-error -shell-escape decoder_only.tex
	open decoder_only.pdf

clean:
	rm *.aux *.dvi *.pdf *.fls *.log *.synctex* *.pyg *.bbl *.brg *.brf *.out *.toc
