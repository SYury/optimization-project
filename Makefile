.SILENT:

.PHONY: theory compile_theory clean_theory test

theory : compile_theory clean_theory

compile_theory :
	pdflatex -shell-escape src/tex/theory.tex

clean_theory :
	rm -f theory.{aux,log}

test :
	echo "No tests available"

