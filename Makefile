all: transformers.tex jcapmod_goon.sty minted.sty bibliography.bib utphys.bst
	pdflatex -shell-escape transformers.tex
	bibtex transformers.aux
	pdflatex -shell-escape transformers.tex
	pdflatex -shell-escape transformers.tex
