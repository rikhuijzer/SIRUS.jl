<div align="center">
    <b>WORK IN PROGRESS</b>
</div>

# StableTrees.jl

This package implements the **S**table and **I**nterpretable **RU**le **S**ets (SIRUS) for classification.
(Regression is also technically possible but not yet implemented.)

The SIRUS algorithm was presented by Bénard et al. in 2020 and 2021.
In short, SIRUS combines the predictive accuracy of random forests with the explainability of decision trees while remaining stable.
Decision trees are easily interpretable but are unstable, meaning that small changes in the dataset can change the model drastically.
Random forests have solved this by fitting multiple trees.
However, interpretability of random forests is limited even with tools such as Shapley values.
For example, it is not possible to reconstruct the model given only the Shapley values.
SIRUS solves these problems by finding 10 to 20 decision rules.

## Algorithm

For more information about the algorithm, see Bénard et al. in 2020 (classification) and 2021 (regression).

### Rule generation

These rules are stable because of a small modification to the original random forest algorithm.
Specifically, the forest structure is restricted.
Next, note that each node of each tree defines a hyperrectangle in the input space.
Each hyperrectangle can be converted to a rule based on whether a query point falls into such the hyperrectangle.
Typically, this process generates 10k rules.

### Rule selection

There is usually large overlap in the rules generated by different trees.
The idea is to deselect those rules that appear less frequent in the finite list of all possible paths.
This deselecting is done via a simple threshold `p₀` on the frequency.
This `p₀` is a hyperparameter of the model.

### Rule set post-treatment

In the post-treatment, all rules are removed which are a linear combination of rules associated with higher frequency paths.
This results in a small set of regression rules.

### Rule aggregation

Next, if a point falls into the corresponding hyperrectangle, then return the average of the outcomes for the training points.
If a point falls outside the corresponding hyperrectangle, then return the average outcome for the points not associated with the hyperrectangle.
Finally, each rule obtains a non-negative weight in order to combine them into a single estimate.
The weights are regularized via Ridge regression since Ridge regression is more stable than Lasso.

for each rule 

## Notes

- Although I fitted the model on correlated data with possibly different distributions (non i.i.d.), the model seems to work quite okay.
    Probably because of Lasso which can handle correlated features in the linear setting.

## TODO

- Let's start by stabilizing the forest structure and verify that that is able to make accurate predictions.
