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

Due to the succesful applications of neural networks in various domains, there has been an paradigm shift within the field of machine learning towards the use of non-interpretable models, also known as "black box" models.
This is appropriate for low stakes domains such as spam detection and product recommendations, but can be problematic in high stakes domains where model decisions have a real-world impact on individuals.
In such situations, black box models may lead to unsafe or unreliable predictions [@doshi2017towards; @barredo2020explainable].
However, the set of fully interpretable models is often limited to linear models and decision trees.
Linear models tend to perform poorly when the data does not satisfy suitable distributions and decision trees perform poorly compared to random forests.
Random forests [@breiman2001random], often outperform linear models and random forests, but are not fully interpretable.
Visualization techniques, such as SHAP [@lundberg2017unified], allow inspection of feature importances, but do not provide enough information to reproduce the predictions made by the model.
The SIRUS algorithm solves these issues by first restricting the split points in the random forest algorithm to a stable subset of points, and by then extracting a small and interpretable rule set [@benard2021interpretable].
However, the original SIRUS algorithm was implemented in C++ and R, which makes it hard to inspect and extend.
An implementation in one high-level language allows verification of the algorithm and allows researchers to investigate further algorithmic improvements.
Furthermore, the original algorithm was covered by a copyleft license meaning that copies are required to be made freely available.
A more permissive license makes it easier to port the algorithm to other languages or move it to production environments.

# Interpretability

To show that the algorithm is fully interpretable, we fitted the model on Haberman's Survival Dataset [@haberman1999survival].
The dataset contains survival data on patients who had undergone surgery for breast cancer and contains three features, namely the number of auxillary `nodes` that were detected, the `age` of the patient at the time of the operation, and the patient's `year` of operation.

```
StableRules model with 8 rules:
 if X[i, :nodes] < 14.0 then 0.152 else 0.094 +
 if X[i, :nodes] < 8.0 then 0.085 else 0.039 +
 if X[i, :nodes] < 4.0 then 0.077 else 0.044 +
 if X[i, :nodes] < 2.0 then 0.071 else 0.047 +
 if X[i, :nodes] < 1.0 then 0.072 else 0.057 +
 if X[i, :year] < 1960.0 then 0.018 else 0.023 +
 if X[i, :age] < 38.0 then 0.029 else 0.023 +
 if X[i, :age] < 42.0 then 0.052 else 0.043
and 2 classes: [0.0, 1.0].
Note: showing only the probability for class 1.0 since class 0.0 has 
      probability 1 - p.
```

This shows that the model contains 8 rules. The first rule, for example, can be explained as: _If the number of detected auxillary nodes is lower than 14, then take 0.152, otherwise take 0.094._

This is done for all 8 rules and the total score is summed to get a prediction.
In essence, the first rule says that if there are less than 14 auxillary nodes detected, then the patient will most likely survive (`class == 1.0`).
In essence, the model states that if there are many auxillary nodes detected, then it's (unfortunately) less likely that the patient will survive.

This model is fully interpretable because there are few rules which can all be interpreted in isolation reasonably well.
Random forests, in contrasts, consist of hundreds to thousands of trees, which are not interpretable due to the large amount of trees.
A common workaround for this is to use SHAP or Shapley values to visualize the fitted model.
The problem with those methods is that they do not allow full reproducibility of the predictions.
For example, if we would inspect the fitted model on the aforementioned Haberman dataset via SHAP, then we could learn feature importances.
In practice that would mean that we could tell which features were important.
In many real-world situations this is not enough.
Imagine having to tell a patient that was misdiagnosed by the model:
"Sorry about our prediction, we were wrong and we didn't really know why.
Only that nodes is an important feature in the model, but we don't know whether this played a large role in your situation."

# Stability

Another problem that the SIRUS algorithm solves is that of model stability.
A stable model is defined as a model which leads to similar conclusions for small changes to data (Yu, 2020).
Unstable models can be difficult to apply in practice since they might require processes to constantly change.
Also, they are considered less trustworthy.

Having said that, most statistical models are quite stable since a higher stability is often correlated to a higher predictive performance.
Put differently, an unstable model by definition leads to different conclusions for small changes to the data and, hence, small changes to the data can cause a sudden drop in predictive performance.
One model which suffers from a low stability is a decision tree. This is because a decision tree will first create the root node of the tree, so a small change in the data can cause the root, and therefore the rest, of the tree to be completely different.
The SIRUS algorithm has solved the instability of random forests by "stabilizing the trees" [@benard2021interpretable] and the authors have proven mathematically that the stabilization works.

# Predictive Performance

# Example

The model can be used via the `MLJ.jl` [@blaom2020mlj] machine learning interface.
For example, this is the code used to fit the model on the full Haberman dataset:

```julia
model = StableRulesClassifier(; max_depth=1, max_rules=8)
mach = machine(model, X, y)
fit!(mach)
```

and model performance was estimated via cross-validation (CV):

```julia
resampling = CV(; nfolds=10, shuffle=true)
evaluate(model, X, y; resampling, measure=auc)
```

# Acknowledgements

This research was supported by the Ministry of Defence, the Netherlands.
We thank Clément Bénard for his help in re-implementing the SIRUS algorithm.
Furthermore, we thank Anthony Bloam and Dávid Hanák (Cursor Insight) for respectively doing code reviews and fixing a critical bug.

# References
