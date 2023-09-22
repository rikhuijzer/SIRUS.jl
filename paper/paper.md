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
affiliations:
 - name: University of Groningen, Groningen, the Netherlands
   index: 1
 - name: Researchable, Assen, the Netherlands
   index: 2
date: '22 September 2023'
bibliography: paper.bib
---

# Summary

[`SIRUS.jl`](https://github.com/rikhuijzer/SIRUS.jl) is a pure Julia implementation of the original Stable and Interpretable RUle Sets (SIRUS) algorithm.
The SIRUS algorithm is a fully interpretable version of random forests, that is, it reduces thousands of trees in the forest to tens of interpretable rules.
With our Julia implementation, we aimed to reproduce the original C++ and R implementation in a high-level language to verify the algorithm as well as making the code easier to read.
We show that the model performs well on binary classification tasks while retaining interpretability and stability.
Furthermore, we made the code available under the permissive MIT license.
In turn, this allows others to research the algorithm further or easily port it to production systems.

# Statement of need

Many of the modern day machine learning models are non-interpretable models, also known as _black box_ models.
For non-interpretable models, predictions cannot be understood by looking at the model parameters.
For example, in Julia, multiple non-interpretable models are available.
One of such models is Flux.jl [@innes2018flux], which provides lightweight abstractions for various advanced models including neural networks.
A more user-friendly alternative is BetaML.jl [@lobianco2021betaml], which provides pure Julia implementations for many models including the perceptron classifier, pegasos classifier [@shalev2007pegasos], neural networks, decision tree classifier and regressor.
Furthermore, an interface to many of Python's packages is available via  ScikitLearn.jl[^3].
Another package is LightGBM.jl, an interface to Microsoft's LightGBM [@ke2017lightgbm]; a gradient boosting machine.
These models are all available via the Machine Learning Julia (MLJ) framework [@blaom2020mlj].
MLJ is the Julia equivalent for Python's scikit-learn [@grisel2023scikit].

For high stakes domains where model decisions have real-world impact on individuals, such as suggesting treatments or personnel selection, these non-interpretable models can be problematic because they may lead to unsafe, unfair, or unreliable predictions [@doshi2017towards; @barredo2020explainable].
However, the set of fully interpretable models is often limited to ordinary and generalized regression models, decision trees, RuleFit, naive Bayes classification, and k-nearest neighbors [@molnar2022interpretable].
In general, linear models can perform poorly when the features are correlated or can be sensitive to the choice of hyperparameters when using regularized models, decision trees perform poorly compared to random forests [@james2013introduction], and RuleFit's performance, anecdotally, was disappointing [@molnar2022interpretable].
Naive Bayes, available in Julia as NaiveBayes.jl[^1], is often overlooked and can be very suitable for many situations; especially when the features are independent [@ashari2013performance].
Alternatively, random forests [@breiman2001random; @sadeghi2022decisiontree] often outperform linear models and decision trees, but are not fully interpretable due to the large number of trees, typically thousands, in the forests.
At the same time, model interpretation techniques, such as SHAP [@lundberg2017unified], do not clearly explain predictions made by the models.
Most interpretation techniques convert the complex model to a simplified representation.
This causes the simplified representation to be different from the complex model and may therefore hide biases and issues related to safety and reliability [@barredo2020explainable].

The SIRUS algorithm solves this problem by simplifying the complex model, and by then using the simplified model for predictions.
This ensures that the same model is used for interpretation and prediction.
However, the original SIRUS algorithm was implemented in about 10k lines of C++ and 2k lines of R[^2] code which makes it hard to inspect and extend due to the combination of two languages.
Our implementation in about 2k lines of Julia code allows researchers to more easily verify the algorithm and investigate further improvements.
Furthermore, the original algorithm was covered by the GPL-3 copyleft license meaning that copies are required to be made freely available.
A more permissive license makes it easier to port the code to other languages or production systems.

[^1]: Source code available at <https://github.com/dfdx/NaiveBayes.jl>.
[^2]: Source code available at <https://gitlab.com/drti/sirus>.
[^3]: Source code available at <https://github.com/cstjean/ScikitLearn.jl>.

# Interpretability

To show that the algorithm is fully interpretable, we fit an example on the Haberman's Survival Dataset [@haberman1999survival].
The dataset contains survival data on patients who had undergone surgery for breast cancer and contains three features, namely the number of axillary _nodes_ that were detected, the _age_ of the patient at the time of the operation, and the patient's _year_ of operation.
For this example, we have set the hyperparameters for the maximum number of rules to 8 since this is a reasonable trade-off between predictive performance and interpretability.
Generally, a higher maximum number of rules will yield a higher predictive performance.
We have also set the maximum depth hyperparameter to 2.
This hyperparameter means that the random forests inside the algorithm are not allowed to have a depth higher than 2.
In turn, this means that rules contain at most 2 clauses (`if A & B`).
When the maximum depth is set to 1, then the rules contain at most 1 clause (`if A`).
Most rule-based models, including SIRUS, are restricted to depth of 1 or 2 [@benard2021sirus].

The output for the fitted model looks as follows (see Section _Code Example_ for the code):

```
StableRules model with 8 rules:
 if X[i, :nodes] < 7.0 then 0.238 else 0.046 +
 if X[i, :nodes] < 2.0 then 0.183 else 0.055 +
 if X[i, :age] ≥ 62.0 & X[i, :year] < 1959.0 then 0.0 else 0.001 +
 if X[i, :year] < 1959.0 & X[i, :nodes] ≥ 2.0 then 0.0 else 0.006 +
 if X[i, :nodes] ≥ 7.0 & X[i, :age] ≥ 62.0 then 0.0 else 0.008 +
 if X[i, :year] < 1959.0 & X[i, :nodes] ≥ 7.0 then 0.0 else 0.003 +
 if X[i, :year] ≥ 1966.0 & X[i, :age] < 42.0 then 0.0 else 0.008 +
 if X[i, :nodes] ≥ 7.0 & X[i, :age] ≥ 42.0 then 0.014 else 0.045
and 2 classes: [0, 1].
```

This shows that the model contains 8 rules where the first rule, for example, can be interpreted as:

_If the number of detected axillary nodes is lower than 7, then take 0.238, otherwise take 0.046._

This calculation is done for all 8 rules and the score is summed to get a prediction.
In essence, the first rule says that if there are less than 8 axillary nodes detected, then the patient will most likely survive (`class == 1.0`).
Put differently, the model states that if there are many axillary nodes detected, then it is, unfortunately, less likely that the patient will survive.
This model is fully interpretable because the model contains a few dozen rules which can all be interpreted in isolation and together.
In contrast, Random forests require hundreds to thousands of trees, which makes them too complex to interpret.

# Stability

Another problem that the SIRUS algorithm addresses is that of model stability.
A stable model is defined as a model which leads to similar conclusions for small changes to data [@yu2020veridical].
Unstable models can be difficult to apply in practice as they might require processes to constantly change.
This also makes such models appear less trustworthy.
Put differently, an unstable model by definition leads to different conclusions for small changes to the data and, hence, small changes to the data could cause a sudden drop in predictive performance.
One model which suffers from a low stability is a decision tree because it will first create the root node of the tree, so a small change in the data can cause the root, and therefore the rest, of the tree to be completely different [@molnar2022interpretable].
Similarly, linear models can be highly sensitive to correlated data and, in the case of regularized linear models, the choice of hyperparameters.
Instead, the SIRUS algorithm provides stability by stabilizing the trees and the authors have proven the correctness of this stabilization mathematically [@benard2021sirus].
In the rest of this paper, we will compare decision trees [@sadeghi2022decisiontree], linear models, XGBoost [@chen2016xgboost], and SIRUS on their predictive performance.
The interpretability and stability are summarized in Table \ref{tab:is}.

\begin{table}[h!]
\small
\centering
\begin{tabular}{|l|c|c|c|c|c|}
\hline
 & \textbf{Decision Tree} & \textbf{Linear Model} & \textbf{XGBoost} & \textbf{XGBoost} & \textbf{SIRUS} \\
& & & \textbf{\scriptsize{max depth: $\mathbb{\infty}$}} & \textbf{\scriptsize{max depth: 2}} & \textbf{\scriptsize{max depth: 2}} \\
& & & & & \textbf{\scriptsize{max rules: 10}} \\
\hline
\textbf{Interpretability} & High & High & Medium & Medium & High \\
\textbf{Stability} & Low & Medium & High & High & High \\
\hline
\end{tabular}
\caption{Summary of interpretability and stability for various models.}
\label{tab:is}
\end{table}

# Predictive Performance

The model is based on random forests and therefore well suited for settings where the number of variables is comparatively large to the number of datapoints [@biau2016random].
To make the random forests interpretable, the large number of trees are converted to a small number of rules.
The conversion works by converting each tree to a set of rules and then pruning the rules by removing simple duplicates and linearly dependent duplicates, see the package documentation or the original paper [@benard2021interpretable] for details.
In practice, this trade-off between between model complexity and interpretability comes at a small performance cost.

To show the performance, we compared SIRUS to a decision tree linear model, and XGBoost; similar to Table \ref{tab:is}.
We have used Julia version 1.9.3 with SIRUS version 1.3.2 (commit `ec9fa73`), 10-fold cross-validation, and we will present variability as $1.96 * \text{standard error}$ for all evaluations with respectively the following datasets, outcome variable type, and measures:
Haberman's Survival Dataset [@haberman1999survival] binary classification dataset with AUC,
Titanic [@eaton1995titanic] binary classification dataset with Area Under the Curve (AUC),
Breast Cancer Wisconsin [@wolberg1995breast] binary classification dataset with AUC,
Pima Indians Diabetes [@smith1988using] binary classification dataset with AUC,
Iris [@fisher1936use] multiclass classification dataset with accuracy,
and Boston Housing [@harrison1978hedonic] regression dataset with $\text{R}^2$; see Table \ref{tab:perf}.
For full details, see [`test/mlj.jl`](https://github.com/rikhuijzer/SIRUS.jl/blob/ec9fa73cb1304de6aa6ce4ec0e116abf59629612/test/mlj.jl).
The output here was copied from the [GitHub Actions Job Summary for commit `ec9fa73`](https://github.com/rikhuijzer/SIRUS.jl/actions/runs/6147235329/attempts/1#summary-16678249371).

\begin{table}[h!]
\small
\centering
\begin{tabular}{|l|c|c|c|c|c|}
\hline
\textbf{Dataset} & \textbf{Decision Tree} & \textbf{Linear Model} & \textbf{XGBoost} & \textbf{XGBoost} & \textbf{SIRUS} \\
& & & \textbf{\scriptsize{max depth: $\mathbb{\infty}$}} & \textbf{\scriptsize{max depth: 2}} & \textbf{\scriptsize{max depth: 2}} \\
& & & & & \textbf{\scriptsize{max rules: 10}} \\
\hline
Haberman & $0.54 \pm 0.07$ & $0.69 \pm 0.06$ & $0.65 \pm 0.04$ & $0.63 \pm 0.04$ & $0.67 \pm 0.06$ \\
Titanic & $0.76 \pm 0.05$ & $0.84 \pm 0.02$ & $0.86 \pm 0.03$ & $0.87 \pm 0.02$ & $0.83 \pm 0.02$ \\
Breast Cancer & $0.92 \pm 0.03$ & $0.98 \pm 0.01$ & $0.99 \pm 0.01$ & $0.99 \pm 0.01$ & $0.98 \pm 0.01$ \\
Diabetes & $0.67 \pm 0.05$ & $0.70 \pm 0.06$ & $0.80 \pm 0.03$ & $0.82 \pm 0.03$ & $0.75 \pm 0.05$ \\
Iris & $0.95 \pm 0.03$ & $0.97 \pm 0.03$ & $0.95 \pm 0.04$ & $0.95 \pm 0.04$ & $0.77 \pm 0.07$ \\
Boston & $0.74 \pm 0.11$ & $0.70 \pm 0.05$ & $0.88 \pm 0.06$ & $0.87 \pm 0.04$ & $0.62 \pm 0.09$ \\
\hline
\end{tabular}
\caption{Predictive performance estimates.}
\label{tab:perf}
\end{table}

At the time of writing, SIRUS's predictive performance is comparable to the linear model and XGBoost on the binary classification datasets, that is, Haberman, Titanic, Breast Cancer, and Diabetes.
The best performance occurs at the Diabetes dataset where both XGBoost and SIRUS outperform the linear model.
The reason for this could be that negative effects are often nonlinear for fragile systems [@taleb2020statistical].
For example, it could be that an increase in oral glucose tolerance increases the chance of diabetes exponentially.
In such cases, the hard cutoff points chosen by tree-based models, such as XGBoost and SIRUS, may fit the data better.

For the multiclass Iris classification and the Boston Housing regression datasets, the performance was worse than the other models.
It could be that this is caused by a bug in the implementation or because this is a fundamental issue in the algorithm.
Further work is needed to find the root cause or workarounds for these low scores.
One possible solution would be to add SymbolicRegression.jl [@cranmer2023interpretable] as a secondary back end for regression tasks.
Similar to SIRUS.jl, SymbolicRegression.jl can fit expressions of a pre-defined form to data albeit with more free parameters, which might fit better but also might cause overfitting, depending on the data.
This achieves performance that is similar to XGBoost [@hanson2023discourse].
In conclusion, the SIRUS algorithm generally performs best on binary classification tasks and can obtain a high predictive performance while retaining model interpretability and stability.

# Code Example

The model can be used via the `MLJ.jl` [@blaom2020mlj] machine learning interface.
For example, this code was used to obtain the fit result for the Haberman example at the start of this paper: <br>
\vspace{2mm}
```julia
using CategoricalArrays: categorical
using CSV: CSV
using DataDeps: DataDeps, DataDep, @datadep_str
using DataFrames
using MLJ
using StableRNGs: StableRNG
using SIRUS: StableRulesClassifier

function register_haberman()
    name = "Haberman"
    message = "Haberman's Survival Data Set"
    remote_path = "https://github.com/rikhuijzer/haberman-survival-dataset/releases/
        download/v1.0.0/haberman.csv"
    checksum = "a7e9aeb249e11ac17c2b8ea4fdafd5c9392219d27cb819ffaeb8a869eb727a0f"
    DataDeps.register(DataDep(name, message, remote_path, checksum))
end

function load_haberman()::DataFrame
    register_haberman()
    path = joinpath(datadep"Haberman", "haberman.csv")
    df = CSV.read(path, DataFrame)
    df[!, :survival] = categorical(df.survival)
    return df
end

data = load_haberman()
X = select(data, Not(:survival))
y = data.survival

model = StableRulesClassifier(; rng=StableRNG(1), q=4, max_depth=2, max_rules=8)

mach = let
    mach = machine(model, X, y)
    MLJ.fit!(mach)
end

mach.fitresult
```
\vspace{2mm}
This code and the output are also available in the [docs](https://sirus.jl.huijzer.xyz/dev/basic-example/).

# Funding

This research was supported by the Ministry of Defence, the Netherlands.

# Acknowledgments

We thank Clément Bénard for his help in re-implementing the SIRUS algorithm.
Furthermore, we thank Anthony Bloam and Dávid Hanák (Cursor Insight) for respectively doing code reviews and finding a critical bug.

# References
