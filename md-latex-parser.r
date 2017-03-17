# Code from https://hopstat.wordpress.com/2015/07/31/rendering-latex-math-equations-in-github-markdown/

library(devtools)
install_github("muschellij2/latexreadme")
library(latexreadme)



#rmd = file.path(script.dir, "README_unparse.md")
rmd <- "/Users/dibya/Documents/Deep Learning papers/deep-learning-paper-summaries/README_unparse.md"
#print(rmd)

args(parse_latex)

#new_md = file.path(script.dir, "README.md")
new_md <- "/Users/dibya/Documents/Deep Learning papers/deep-learning-paper-summaries/README.md"
parse_latex(rmd,
            new_md,
            git_username = "dibyatanoy",
            git_reponame = "deep-learning-paper-summary")

