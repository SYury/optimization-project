build : compile clean

compile :
	pdflatex -shell-escape src/tex/theory.tex

clean :
	rm -f theory.{aux,log}
