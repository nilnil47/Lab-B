#jupyter-nbconvert --to html magnetise.ipynb
#pandoc magnetise.html -t latex --pdf-engine=pdflatex -o magnetism.pdf

jupyter nbconvert --no-input --output-dir=. magnetise.ipynb --to pdf