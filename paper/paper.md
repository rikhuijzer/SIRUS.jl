---
title: 'SIRUS.jl: Interpretable Machine Learning via Rule Extraction'
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
date: '29 July 2023'
bibliography: paper.bib
---

# Summary

`SIRUS.jl` is a pure Julia implementation of the original Stable and Interpretable RUle Sets (SIRUS) algorithm.
The SIRUS algorithm is a fully interpretable version of random forests, that is, it reduces the large amount of trees in the forest to a small amount of interpretable rules.
With our Julia implementation, we aimed to reproduce the original C++ and R implementation in a high-level language to verify the algorithm as well as making the code easier to read.
Furthermore, we made the code available under the permissive MIT license.
In turn, this allows others to research the algorithm further or easily port it to production environments.

# Statement of need

Due to the succesful applications of neural networks in various domains, the field of machine learning is shifting more and more towards non-interpretable black box models.
This is appropriate for low stakes domains such as spam detection and product recommendations, but can be problematic in high stakes domains where model decisions have a real-world impact on individuals.
In such situations, non-interpretable models may lead to unsafe or unreliable predictions [@doshi2017towards; @barredo2020explainable].
However, the set of fully interpretable models is often limited to linear models and decision trees.
Linear models tend to perform poorly when the data does not satisfy suitable distributions and decision trees perform poorly compared to random forests.
Random forests [@breiman2001random], perform very well in many machine learning tasks [@biau2016random], but are not fully interpretable.
Visualization techniques, such as SHAP [@lundberg2017unified], allow inspection of feature importances, but do not provide enough information to reproduce the predictions made by the model.
The SIRUS algorithm solves these problems by first restricting the split points in the random forest algorithm to a stable subset of points, and by then extracting a small and interpretable rule set [@benard2021interpretable].
However, this original SIRUS algorithm was implemented in C++ and R, which makes it hard to inspect and extend.
An implementation in one high-level language allows verification of the algorithm and allows researchers to investigate further algorithmic improvements.
Furthermore, the original algorithm was covered by a copyleft license meaning that copies are required to be made freely available.
A more permissive license would make it easier to port the algorithm to other languages or move it to production environments.

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

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

This research was supported by the Ministry of Defence, the Netherlands.
We thank Clément Bénard for his help in re-implementing the SIRUS algorithm.
Furthermore, we thank Anthony Bloam and Dávid Hanák (Cursor Insight) for respectively doing code reviews and fixing a critical bug.

# References
