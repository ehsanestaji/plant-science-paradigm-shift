.PHONY: all clean

MAIN = main

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib sections/*.tex
	pdflatex $(MAIN)
	bibtex $(MAIN)
	pdflatex $(MAIN)
	pdflatex $(MAIN)

clean:
	rm -f $(MAIN).{aux,bbl,blg,log,out,toc,pdf}
	rm -f sections/*.aux
