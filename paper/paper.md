---
title: 'SIRUS.jl: Pure Julia implementation of the Stable and Interpretable RUle Sets (SIRUS) algorithm '
authors:
  - name: 'Rik Huijzer'
    corresponding: true
    orcid: '0000-0001-9445-8466'
    affiliation: '1'
  - name: 'Frank Blaauw'
    affiliation: '2'
    orcid: '0000-0002-6588-5079'
  - name: 'Ruud J.R. den Hartigh'
    affiliation: '1'
    orcid: '0000-0002-0094-8307'
  - name: 'Peter de Jonge'
    affiliation: '1'
    orcid: '0000-0002-0866-6929'
affiliations:
 - name: University of Groningen, Groningen, the Netherlands
   index: 1
 - name: Researchable, Assen, the Netherlands
   index: 2
date: '06 July 2023'
bibliography: paper.bib
---

**Note-to-self: Focus on the implementation and keep this a 2-3 page document, in line with other JOSS papers.**

# Summary

`SIRUS.jl` is a pure Julia implementation of the original Stable and Interpretable RUle Sets (SIRUS) algorithm.
The SIRUS algorithm is a fully interpretable version of random forests, that is, it reduces the large amount of trees in the forest to a small amount of interpretable rules.
With our Julia implementation, we aimed to reproduce the original C++ and R implementation in a high-level language to verify the algorithm as well as making the code easier to read.
In turn, this allows others to build upon this Julia implementation or port the algorithm to production environments.

# Statement of need

The random forest algorithm [@breiman2001random] archieves high predictive performance for many machine learning tasks; especially when the number of observations is low relative to the number of variables [@biau2016random].
The algorithm does this by fitting a large number of trees.
However, it can be difficult to fully interpret the model predictions since that would require inspecting all trees.
Visualization techniques, such as SHAP [@lundberg2017unified], alleviate this problem but do not fully solve it.
SHAP does allow for inspecting feature importances, but does not allow each prediction to be reproduced from the interpretable information alone.
This may lead to unsafe or unreliable predictions [@doshi2017towards, @barredo2020explainable].
The SIRUS algorithm was created to solve this problem and does so by first restricting the split points in the random forest algorithm to a stable subset of points, and by then extracting a small and interpretable rule set [@benard2021interpretable].
This original algorithm was implemented in C++ and R, which makes it hard to read.
A pure Julia implementation allows the results to be reproduced and makes the code easier to read.
In turn, this could allow others to research further algorithmic improvements or to port the algorithm to production environments.

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Clément Bénard for his help in re-implementing the SIRUS algorithm.
Also, we acknowledge contributions from Hylke Cornelis Donker for discussions about the analysis and for his help in the visualizations.
Furthermore, Anthony Bloam and Dávid Hanák (Cursor Insight) contributed by respectively doing code reviews and by finding a critical bug.

# References
