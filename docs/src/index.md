# SIRUS

This package is a pure Julia implementation of the **S**table and **I**nterpretable **RU**le **S**ets (SIRUS) algorithm.
The algorithm was originally created by Clément Bénard, Gérard Biau, Sébastien Da Veiga, and Erwan Scornet (Bénard et al., [2021](http://proceedings.mlr.press/v130/benard21a.html)).
`SIRUS.jl` has implemented both classification and regression.
For R users, the original version of the SIRUS algorithm is available via [CRAN](https://cran.r-project.org/web/packages/sirus/index.html).
Compared to the R version, this Julia implementation is more easy to inspect than the original R and C++ implementation.
Furthermore, this implementation is fast and integrated with the `MLJ.jl` machine learning ecosystem.
With this, multiple benchmarks are executed and checked with every test run.
The results are listed below each GitHub Actions run.

The algorithm is based on random forests.
Random forests perform generally very well; especially on datsets with a relatively high number of features compared to the number of datapoints (Biau & Scornet, [2016](https://doi.org/10.1007/s11749-016-0481-7)).
However, random forests are hard to interpret because of the large number of, sometimes large, trees.
Interpretability methods such as SHAP alleviate this problem slightly, but still do not fully explain predictions.
Put differently, it is not possible to reproduce predictions on the feature importances that SHAP reports.
SIRUS solved this by converting the large number of trees to interpretable rules.
These rules fully explain the predictions while remaining easy to interpret.

## Where to Start?

- [Binary Classification Example](/dev/binary-classification)

## Acknowledgements

Thanks to Clément Bénard, Gérard Biau, Sébastian da Veiga and Erwan Scornet for creating the SIRUS algorithm and documenting it extensively.
Special thanks to Clément Bénard for answering my questions regarding the implementation.
Thanks to Hylke Donker for figuring out a way to visualize these rules.
Also thanks to my PhD supervisors Ruud den Hartigh, Peter de Jonge and Frank Blaauw, and Age de Wit and colleagues at the Dutch Ministry of Defence for providing the data clarifying the constraints of the problem and for providing many methodological suggestions.
