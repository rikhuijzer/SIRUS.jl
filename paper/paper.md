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
  - name: 'Peter de Jonge'
    affiliation: '1'
    orcid: '0000-0002-0866-6929'
  - name: 'Ruud J.R. den Hartigh'
    affiliation: '1'
    orcid: '0000-0002-0094-8307'
affiliations:
 - name: University of Groningen, Groningen, the Netherlands
   index: 1
 - name: Researchable, Assen, the Netherlands
   index: 2
date: '29 July 2023'
bibliography: paper.bib
---

# Summary

[`SIRUS.jl`](https://github.com/rikhuijzer/SIRUS.jl) is a pure Julia implementation of the original Stable and Interpretable RUle Sets (SIRUS) algorithm.
The SIRUS algorithm is a fully interpretable version of random forests, that is, it reduces thousands of trees in the forest to a tens of interpretable rules.
With our Julia implementation, we aimed to reproduce the original C++ and R implementation in a high-level language to verify the algorithm as well as making the code easier to read.
Furthermore, we made the code available under the permissive MIT license.
In turn, this allows others to research the algorithm further or easily port it to production systems.

# Statement of need

Many of the modern day machine learning models are non-interpretable models, also known as _black box_ models.
These models can be problematic in high stakes domains where model decisions have real-world impact on individuals.
In such situations, black box models may lead to unsafe or unreliable predictions [@doshi2017towards; @barredo2020explainable].
However, the set of fully interpretable models is often limited to linear models and decision trees.
Linear models tend to perform poorly when the data does not satisfy suitable distributions and decision trees perform poorly compared to random forests [@james2013introduction].
Instead, random forests [@breiman2001random] often outperform linear models and decision trees, but are not fully interpretable.
At the same time, visualization techniques, such as SHAP [@lundberg2017unified], allow inspection of feature importances, but do not provide enough information to reproduce the predictions made by the model.
The SIRUS algorithm solves these issues by first restricting the split points in the random forest algorithm to a stable subset of points, and by then extracting a small and interpretable rule set [@benard2021interpretable].
However, the original SIRUS algorithm was implemented in C++ and R, which makes it hard to inspect and extend due to the combination of two languages.
An implementation in one high-level language allows verification of the algorithm and allows researchers to investigate further algorithmic improvements.
Furthermore, the original algorithm was covered by a copyleft license meaning that copies are required to be made freely available.
A more permissive license makes it easier to port the code to other languages or production systems.

# Interpretability

To show that the algorithm is fully interpretable, we fitted the model on Haberman's Survival Dataset [@haberman1999survival].
The dataset contains survival data on patients who had undergone surgery for breast cancer and contains three features, namely the number of auxillary `nodes` that were detected, the `age` of the patient at the time of the operation, and the patient's `year` of operation.

```
StableRules model with 8 rules:
 if X[i, :nodes] < 8.0 then 0.156 else 0.031 +
 if X[i, :nodes] < 14.0 then 0.164 else 0.026 +
 if X[i, :nodes] < 4.0 then 0.128 else 0.037 +
 if X[i, :nodes] ≥ 8.0 & X[i, :age] < 38.0 then 0.0 else 0.008 +
 if X[i, :year] ≥ 1966.0 & X[i, :age] < 42.0 then 0.0 else 0.005 +
 if X[i, :nodes] < 2.0 then 0.107 else 0.034 +
 if X[i, :year] ≥ 1966.0 & X[i, :age] < 38.0 then 0.0 else 0.001 +
 if X[i, :year] < 1959.0 & X[i, :nodes] ≥ 2.0 then 0.0 else 0.003
and 2 classes: [0.0, 1.0].
Note: showing only the probability for class 1.0 since class 0.0 has
      probability 1 - p.
```

This shows that the model contains 8 rules. The first rule, for example, can be interpreted as:
_If the number of detected auxillary nodes is lower than 8, then take 0.156, otherwise take 0.031._

This is done for all 8 rules and the total score is summed to get a prediction.
In essence, the first rule says that if there are less than 8 auxillary nodes detected, then the patient will most likely survive (`class == 1.0`).
Put differently, the model states that if there are many auxillary nodes detected, then it is (unfortunately) less likely that the patient will survive.

This model is fully interpretable because there are few rules which can all be interpreted in isolation reasonably well.
Random forests, in contrasts, consist of hundreds to thousands of trees, which are not interpretable due to this large number.
A common workaround for this is to use SHAP or Shapley values to visualize the fitted model.
The problem with those methods is that they do not allow full reproducibility of the predictions.
For example, if we would inspect the fitted model on the aforementioned Haberman dataset via SHAP, then we could learn feature importances.
In practice that would mean that we could tell which features were important.
In many real-world situations this is not enough.
For example, when using only feature importances, it would be unclear for a doctor how the prediction for a specific patient was made.
The doctor would only only know that some features are in general more important than other features.

# Stability

Another problem that the SIRUS algorithm solves is that of model stability.
A stable model is defined as a model which leads to similar conclusions for small changes to data [@yu2020veridical].
Unstable models can be difficult to apply in practice as they might require processes to constantly change.
This also makes such models appear less trustworthy.
Put differently, an unstable model by definition leads to different conclusions for small changes to the data and, hence, small changes to the data could cause a sudden drop in predictive performance.
One model which suffers from a low stability is a decision tree because it will first create the root node of the tree, so a small change in the data can cause the root, and therefore the rest, of the tree to be completely different [@molnar2022interpretable].
The SIRUS algorithm has solved the instability of random forests by "stabilizing the trees" and the authors have proven the correctness of this stabilization mathematically [@benard2021interpretable].

# Predictive Performance

The model is based on random forests and therefore has good performance in settings where the number of variables is comparatively large to the number of datapoints [@biau2016random].
The algorithm converts a large number of trees to a small number of rules to improve interpretability.
This tradeoff between model complexity and interpretability comes at a small performance cost.

To evaluate the performance of SIRUS, we have compared it to a linear model, a decision tree [@sadeghi2022decisiontree], and the XGBoost [@chen2016xgboost] gradient boosting algorithm.
We have used SIRUS.jl version 1.2.1, 10-fold cross-validation and present variability as $1.96 * \text{standard error}$ for all evaluations with respectively the following datasets and measures:
Titanic [@eaton1995titanic] with Area Under the Curve (AUC),
Breast Cancer Wisconsin [@wolberg1995breast] with accuracy,
Haberman's Survival Dataset [@haberman1999survival] with AUC,
Iris [@fisher1936use] with accuracy,
and Boston Housing [@harrison1978hedonic] with $\text{R}^2$; see Table \ref{tab:perf}.

\begin{table}[h!]
\small
\centering
\begin{tabular}{|l|l|c|c|c|c|c|c|}
\hline
& \textbf{Max} & & \textbf{Breast} & & & \textbf{Boston} \\
\textbf{Model} & \textbf{depth} & \textbf{Titanic} & \textbf{Cancer} & \textbf{Haberman} & \textbf{Iris} & \textbf{Housing} \\
\hline
Linear & & $0.84 \pm 0.02$ & $0.93 \pm 0.03$ & $0.69 \pm 0.06$ & $0.97 \pm 0.03$ & $0.70 \pm 0.05$ \\
Decision Tree & & $0.75 \pm 0.05$ & $0.92 \pm 0.02$ & $0.54 \pm 0.06$ & $0.95 \pm 0.03$ & $0.71 \pm 0.11$ \\
XGBoost & & $0.86 \pm 0.03$ & $0.96 \pm 0.02$ & $0.65 \pm 0.04$ & $0.95 \pm 0.04$ & $0.88 \pm 0.06$ \\
XGBoost & 2 & $0.87 \pm 0.02$ & $0.96 \pm 0.02$ & $0.63 \pm 0.04$ & $0.94 \pm 0.04$ & $0.87 \pm 0.04$ \\
SIRUS & 2 & $0.82 \pm 0.02$ & $0.93 \pm 0.02$ & $0.67 \pm 0.07$ & $0.71 \pm 0.08$ & $0.63 \pm 0.10$ \\
\hline
\end{tabular}
\caption{Predictive performance estimates over 10-fold cross-validation for a linear model, decision tree, XGBoost, and SIRUS on various public datasets.}
\label{tab:perf}
\end{table}

This shows that the SIRUS algorithm performs very comparable to the state-of-the-art LGBM classifier by Microsoft.
The tree depths are set to at most 2 because rules which belong to a depth of 3 will (almost) never show up in the final model.

# Code Example

The model can be used via the `MLJ.jl` [@blaom2020mlj] machine learning interface.
For example, this is the code used to fit the model on the full Haberman dataset: <br>
\vspace{2mm}
```julia
model = StableRulesClassifier(; max_depth=2, max_rules=8)
mach = machine(model, X, y)
fit!(mach)
```
\vspace{2mm}
and model performance was estimated via cross-validation (`CV`):
\vspace{2mm}
```julia
resampling = CV(; nfolds=10, shuffle=true)
evaluate(model, X, y; resampling, measure=auc)
```

# Funding

This research was supported by the Ministry of Defence, the Netherlands.

# Acknowledgements

We thank Clément Bénard for his help in re-implementing the SIRUS algorithm.
Furthermore, we thank Anthony Bloam and Dávid Hanák (Cursor Insight) for respectively doing code reviews and finding a critical bug.

# References
