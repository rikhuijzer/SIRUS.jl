### A Pluto.jl notebook ###
# v0.19.26

using Markdown
using InteractiveUtils

# ╔═╡ 7c10c275-54d8-4f1a-947f-7861199cdf21
# ╠═╡ show_logs = false
# hideall
begin
	ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

	PKGDIR = dirname(dirname(@__DIR__))
    DOCS_DIR = dirname(@__DIR__)
	using Pkg: Pkg
	Pkg.activate(DOCS_DIR)
	Pkg.develop(; path=PKGDIR)
end

# ╔═╡ f833dab6-31d4-4353-a68b-ef0501d606d4
begin
	ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

	using CairoMakie
	using CategoricalArrays: categorical
	using CSV: CSV
	using DataDeps: DataDeps, DataDep, @datadep_str
	using DataFrames
	using DecisionTree: DecisionTree
	using LightGBM.MLJInterface: LGBMClassifier
	using MLJDecisionTreeInterface: DecisionTreeClassifier
	using MLJ: CV, MLJ, Not, PerformanceEvaluation, auc, fit!, evaluate, machine
	using PlutoUI: TableOfContents # hide
	using StableRNGs: StableRNG
	using SIRUS
	using Statistics: mean, std
end

# ╔═╡ e9028115-d098-4c61-a82f-d4553fe654f8
# hideall
TableOfContents()

# ╔═╡ b1c17349-fd80-43f1-bbc2-53fdb539d1c0
md"""
This page will provide an overview of the algorithm and describe how it works and how it can be used.
To do this, let's start by briefly describing random forests.
"""

# ╔═╡ 348d1235-87f2-4e8f-8f42-be89fef5bf87
md"""
## Random forests

Random forests are known to produce accurate predictions especially in settings where the number of features `p` is close to or higher than the number of observations `n` (Biau & Scornet, [2016](https://doi.org/10.1007/s11749-016-0481-7)).
Let's start by explaining the building blocks of random forests: decision trees.
As an example, we take Haberman's Survival Data Set (see the _Appendix_ below for more code details):
"""

# ╔═╡ 4c8dd68d-b193-4846-8d93-ab33512c3fa2
md"""
This dataset contains observations from a study with patients who had breast cancer.
The `survival` column contains a `0` if a patient has died within 5 years and `1` if the patient has survived for at least 5 years.
The aim is to predict survival based on the `age`, the `year` in which the operation was conducted and the number of detected auxillary `nodes`.
"""

# ╔═╡ f75aa57f-6e84-4f7e-88e4-11a00cb9ad2b
md"""
Via [`MLJ.jl`](https://github.com/alan-turing-institute/MLJ.jl), we can fit multiple decision trees on this dataset:
"""

# ╔═╡ e5a45b1a-d761-4279-834b-216df2a1dbb5
md"""
This has fitted various trees to various subsets of the dataset via cross-validation.
Here, I've set `max_depth=2` to simplify the fitted trees which makes the tree more easily explainable.
Also, for our small dataset, this forces the model to remain simple so it likely reduces overfitting.
Let's look at the first tree:
"""

# ╔═╡ d38f8814-c7b8-4911-9c63-d99b646b4486
md"""
What this shows is that the first tree decided that the `nodes` feature is the most helpful in deciding who will survive for 5 more years.
Next, if the `nodes` feature is below 2.5, then `age` will be selected on.
If `age < 79.5`, then the model will predict the second class and if `age ≥ 79.5` it will predict the first class.
Similarly for `age < 43.5`.
Now, let's see what happens for a slight change in the data.
In other words, let's see how the fitted model for the second split looks:
"""

# ╔═╡ 5318414e-5c87-4be0-bcd0-b6efd4eee5b9
md"""
This shows that the features and the values for the splitpoints are not the same for both trees.
This is called stability.
Or in this case, a decision tree is considered to be unstable.
This instability is problematic in situations where real-world decisions are based on the outcome of the model.
Imagine using this model for the selecting which students are allowed to enter some university.
If the model is updated every year with the data from the last year, then the selection criteria would vary wildly per year.
This instability also causes accuracy to fluctuate wildly.
Intuitively, this makes sense: if the model changes wildly for small data changes, then model accuracy also changes wildly.
This intuitively also implies that the model is more likely to overfit.
This is why random forests were introduced.
Basically, random forests fit a large number of trees and average their predictions to come to a more accurate prediction.
The individual trees are obtained by restricting the observations and the features that the trees are allowed to use.
For the restriction on the observations, the trees are only allowed to see `partial_sampling * n` observations.
In practise, `partial_sampling` is often 0.7.
The restriction on the features is defined in such a way that it guarantees that not every tree will take the same split at the root of the tree.
This makes the trees less correlated (James et al., [2021](https://doi.org/10.1007/978-1-0716-1418-1); Section 8.2.2) and, hence, more accurate.

Unfortunately, these random forests are hard to interpret.
To interpret the model, individuals would need to interpret hundreds to thousands of trees containing multiple levels.
Alternatively, methods have been created to visualize these uninterpretable models (for example, see Molnar ([2022](https://christophm.github.io/interpretable-ml-book/)); Chapters 6, 7 and 8).
The most promising one of these methods are Shapley values and SHAP.
These methods show which features have the highest influence on the prediction.
See my blog post on [Random forests and Shapley values](https://huijzer.xyz/posts/shapley/) for more information.
Knowing which features have the highest influence is nice, but they do not state exactly what feature is used and at what cutoff.
Again, this is not good enough for selecting students into universities.
For example, what if the government decides to ask for details about the selection?
The only answer that you can give is that some features are used for selection more than others and that they are on average used in a certain direction.
If the government asks for biases in the model, then these are impossible to report.
In practice, the decision is still a black-box.
SIRUS solves this by extracting easily interpretable rules from the random forests.
"""

# ╔═╡ d816683b-2f7d-45a7-bd40-42f554a48b1b
md"""
## Rule-based models

Rule-based models promise much greater interpretability than random forests.
Instead of returning a large number of trees, rule-based models return a set of rules.
Each rule can be interpreted on its own and the final model aggregates these rules by summing the prediction of each rules.
For example, one rule can be:

> if `nodes < 4.5` then chance of survival is 0.6 and if `nodes ≥ 4.5` then chance of survival is 0.4.

Note that these rules can be extracted quite easily from the decision trees.
For splits on the second level of the tree, the rule could look like:

> if `nodes < 4.5` and `age < 38.5` then chance of survival is 0.8 and otherwise the chance of survival is 0.4.

When applying this extracting of rules to a random forest, there will be thousands of rules.
Next, via some heuristic, the most important rules can be localized and these rules then result in the final model.
See, for example, RuleFit (Friedman & Popescu, [2008](https://www.jstor.org/stable/30245114)).
The problem with this approach is that they are fitted on the unstable decision trees that were shown above.
As an example, on time the tree splits on `age < 43.5` and another time on `age < 44.5`.
"""

# ╔═╡ 4b67c47a-ee98-495e-bb1b-41db83c11cd4
md"""
## Tree stabilization

In the papers which introduce SIRUS, Bénard et al. ([2021a](https://doi.org/10.1214/20-EJS1792), [2021b](https://proceedings.mlr.press/v130/benard21a.html)) proof that their algorithm is stable and that the other algorithms are not.
They achieve their stability by restricting the location at which the splitpoints can be chosen.
To see how this works, let's look at the `age` feature on its own.
"""

# ╔═╡ 0d121fa3-fbfa-44e5-904b-64a1622ec91b
md"""
The default random forest algorithm is allowed to choose any location inside this feature to split on.
To avoid having to figure out locations by itself, the algorithm will choose on of the datapoints as a split location.
So, for example, the following split indicated by the red vertical line would be a valid choice:
"""

# ╔═╡ 896e00dc-2ce9-4a9f-acc1-519aec21dd83
md"""
But what happens if we take a random subset of the data?
Say, we take the following subset of length `0.7 * length(nodes)`:
"""

# ╔═╡ ee12350a-627b-4a11-99cb-38c496977d18
md"""
Now, the algorithm would choose a different location and, hence, introduce instability.
To solve this, Bénard et al. decided to limit the splitpoints that the algorithm can use to split to data to a pre-defined set of points.
For each feature, they find `q` empirical quantiles where `q` is typically 10.
Let's overlay these quantiles on top of the `age` feature:
"""

# ╔═╡ 0cc970cd-b7ed-4782-a520-ff0a76fe0453
md"""
Next, let's see where the cutpoints are when we take the same random subset as above:
"""

# ╔═╡ 01b08d44-4b9b-42e2-bb20-f34cb9b407f3
md"""
As can be seen, many cutpoints are at the same location as before.
Furthermore, compared to the unrestricted range, the chance that two different trees who see a different random subset of the data will select the same cutpoint has increased dramatically.

The benefit of this is that it is now quite easy to extract the most important rules.
Rule extraction consists of simplifying them a bit and ordering them by frequency of occurrence.
Let's see how accurate this model is.
"""

# ╔═╡ 7e1d46b4-5f93-478d-9105-a5b0db1eaf08
md"""
## Benchmarks

Let's compare the following models:

- Decision tree (`DecisionTreeClassifier`)
- Stabilized random forest (`StableForestClassifier`)
- SIRUS (`StableRulesClassifier`)
- LightGBM (`LGBMClassifier`)

The latter is a state-of-the-art gradient boosting model created by Microsoft.
See the Appendix for more details about these results.
"""

# ╔═╡ 4a4ab7ef-659e-4048-ab16-94ad4cb4328a
md"""
As can be seen, the score of the stabilized random forest (`StableForestClassifier`) is almost as good as Microsoft's state-of-the-art classifier (`LGBMClassifier`), but both are not interpretable since that requires interpreting thousands of trees.
With the rule-based classifier (`StableRulesClassifier`), a small amount of predictive performance can be traded for high interpretability.
Note that the rule-based classifier may actually be more accurate in practice because verifying and debugging the model is much easier.

Regarding the hyperparameters, tuning `max_rules` and `max_depth` has the most effect.
`max_rules` specifies the number of rules to which the random forest is simplified.
Setting to a high number such as 999 makes the predictive performance similar to that of a random forest, but also makes the interpretability as bad as a random forest.
Therefore, it makes more sense to truncate the rules to somewhere in the range 5 to 40 to obtain accurate models with high interpretability.
`max_depth` specifies how many levels the trees have.
For larger datasets, `max_depth=2` makes the most sense since it can find more complex patterns in the data.
For smaller datasets, `max_depth=1` makes more sense since it reduces the chance of overfitting.
It also simplifies the rules because with `max_depth=1`, the rule will contain only one conditional (for example, "if A then ...") versus two conditionals (for example, "if A & B then ...").
In some cases, model accuracy can be improved by increasing `n_trees`.
The higher this number, the more trees are fitted and, hence, the higher the chance that the right rules are extracted from the trees.
"""

# ╔═╡ 16de5518-2a16-40ef-87a5-d2acd514d294
md"""
## Interpretation

Finally, let's interpret the rules that the model has learned.
Since we know that the model performs well on the cross-validations, we can fit our preferred model on the complete dataset:
"""

# ╔═╡ 6d0b29b6-61fb-4d16-9389-071892a3d9db
md"""
The interpretation of the fitted model is as follows.
The model has learned three rules for this dataset.
For making a prediction for some value at row `i`, the model will first look at the value for the `nodes` feature.
If the value is below the listed number, then the number after `then` is chosen and otherwise the number after `else`.
This is done for all the rules and, finally, the rules are summed to obtain the final prediction.
"""

# ╔═╡ 3c415a26-803e-4f35-866f-2e582c6c1c45
md"""
## Visualization

Since our rules are relatively simple with only a binary outcome and only one clause in each rule, the following figure is a way to visualize the obtained rules per fold.
For multiple clauses, I would not know how to visualize the rules.
Also, this plot is probably not perfect; let me know if you have suggestions.

This figure shows the model uncertainty.
The x-position on the left shows `log(else-scores / if-scores)`, the vertical lines on the right show the threshold, and the histograms on the right show the data. 
For example, for the `nodes`, it can be seen that all rules (fitted in the different cross-validation folds) base their decision on whether the `nodes` are below, roughly, 5. 
Next, the left side indicates that the individuals who had less than 5 nodes are more likely to survive, according to the model. 
The sizes of the dots indicate the weight that the rule has, so a bigger dot means that a rule plays a larger role in the final outcome. 
These dots are sized in such a way that a doubling in weight means a doubling in surface size. 
Finally, the variables are ordered by the sum of the weights.
"""

# ╔═╡ ab5423cd-c8a9-488e-9bb0-bb41e583c2fa
md"""
What this plot shows is that the `nodes` feature is on average chosen as the feature with the most predictive power.
This can be concluded because the `nodes` feature is shown as the first feature and the tickness of the dots is the biggest.
Furthermore, there is agreement on the effect of the `nodes` and `age` features.
In both cases, a lower number is associated with survival.
This is as expected because the model essentially implies that people where less cancerous auxillary nodes are detected and who are younger are more likely to survive.
The `year` in which the operation was conducted shouldn't have serious effect on the survivability and the model shoes this by a high variability on that feature.
"""

# ╔═╡ f2fee9a8-7f6f-4213-9046-2f1a8f14a7e6
md"""
## Practical applications

As shown in the previous sections, the model satisfies two things:

1. It shows a good predictive performance in the model evaluations. The performance is slightly lower than more complex models, but this tradeoff can be worth it because the rule-based model is interpretable.
2. The fitted model makes theoretical sense. As shown in the visualization, the `nodes` and `age` features are the most important for prediction and both features are used in the expected way.

Since the model shows good performance and makes theoretical sense, we can be reasonably sure that the model will generalize to new data in a similar context.
Next, the model can be applied by fitting it on the full dataset and brining it to a real-world setting.

Note that unlike the state-of-the-art random forest from Microsoft, each decision that the model makes can be fully explained.
All rules can be read stand-alone and interpreted.
For example, when trying to interpret a random forest, it will only report feature importances.
For the Haberman dataset, we would know more than `nodes` is negatively associated and `age` too.
With the rule-based model, we can say exactly at which number of `nodes` and at which `age` the model decides to split the data between likely to survive or not survive.
"""

# ╔═╡ e6b880e9-e263-4818-81e9-bb4105e5c2c1
md"""
## Conclusion

Compared to decision trees, the rule-based classifier is more stable, more accurate and similarly easy to interpet.
Compared to the random forest, the rule-based classifier is only slightly less accurate, but much easier to interpet.
Due to the interpretability, it is likely easier to verify the model and therefore the rule-based classifier will be more accurate in real-world settings.
This makes rule-based highly suitable for many machine learning tasks.
"""

# ╔═╡ cfd908a0-1ee9-461d-9309-d4ffe738ba8e
# hideall
function _threshold(rule)
	sp = only(rule.path.splits).splitpoint
	return sp.value
end;

# ╔═╡ e7f396dc-38a7-40f7-9e5b-6fbea9d61789
# hideall
function _rotation(left, right)
	# π/2 points to the left and -(π/2) points to the right.
	if right < left
		return (left - right) / left * (π/2)
	else
		return (right - left) / right * (-π/2)
	end
end;

# ╔═╡ 7c688412-d1b4-492d-bda2-0b9181057d4d
# hideall
function _rule_index(model::SIRUS.StableRules, feature_name::String)
	for (i, rule) in enumerate(model.rules)
		if only(rule.path.splits).splitpoint.feature_name == feature_name
			return i
		end
	end
	return nothing
end;

# ╔═╡ aa93a6c4-d5a0-4c73-9db9-e26c3c3f526b
# hideall
function _sum_weights(fitresults::Vector, name::AbstractString)
	indexes = _rule_index.(fitresults, Ref(name))
	return sum([isnothing(index) ? 0 : fitresults[i].weights[index] for (i, index) in enumerate(indexes)])
end;

# ╔═╡ e1890517-7a44-4814-999d-6af27e2a136a
md"""
## Appendix
"""

# ╔═╡ ede038b3-d92e-4208-b8ab-984f3ca1810e
function _plot_cutpoints(data::AbstractVector)
	fig = Figure(; resolution=(800, 100))
	ax = Axis(fig[1, 1])
	cps = Float64.(unique(cutpoints(data, 10)))
	scatter!(ax, data, fill(1, length(data)))
	vlines!(ax, cps; color=:black, linestyle=:dash)
	textlocs = [(c, 1.1) for c in cps]
	for cutpoint in cps
		annotation = string(round(cutpoint; digits=2))::String
		text!(ax, cutpoint + 0.2, 1.08; text=annotation, fontsize=13)
	end
	ylims!(ax, 0.9, 1.2)
	hideydecorations!(ax)
	return fig
end;

# ╔═╡ 93a7dd3b-7810-4021-bf6e-ae9c04acea46
_rng(seed::Int=1) = StableRNG(seed);

# ╔═╡ be324728-1b60-4584-b8ea-c4fe9e3466af
function _io2text(f::Function)
	io = IOBuffer()
	f(io)
	s = String(take!(io))
	return Base.Text(s)
end;

# ╔═╡ 7ad3cf67-2acd-44c6-aa91-7d5ae809dfbc
function _evaluate(model, X, y; nfolds=10)
    resampling = CV(; nfolds, shuffle=true, rng=_rng())
    acceleration = MLJ.CPUThreads()
    evaluate(model, X, y; acceleration, verbosity=0, resampling, measure=auc)
end;

# ╔═╡ 0ca8bb9a-aac1-41a7-b43d-314a4029c205
# hideall
S = SIRUS;

# ╔═╡ 9db18ac7-4508-4861-8854-3e19d5218309
function register_haberman()
    name = "Haberman"
    message = "Slightly modified copy of Haberman's Survival Data Set"
    remote_path = "https://github.com/rikhuijzer/haberman-survival-dataset/releases/download/v1.0.0/haberman.csv"
    checksum = "a7e9aeb249e11ac17c2b8ea4fdafd5c9392219d27cb819ffaeb8a869eb727a0f"
    DataDeps.register(DataDep(name, message, remote_path, checksum))
end;

# ╔═╡ 564598d7-5bdf-4e42-b812-8aca20fa20d4
function _haberman()
	register_haberman()
    dir = datadep"Haberman"
    path = joinpath(dir, "haberman.csv")
    df = CSV.read(path, DataFrame)
    df[!, :survival] = categorical(df.survival)
    # Need Floats for the LGBMClassifier.
    for col in [:age, :year, :nodes]
        df[!, col] = float.(df[:, col])
    end
    return df
end;

# ╔═╡ 961aa273-d97b-497f-a79a-06bf89dc34b0
data = _haberman()

# ╔═╡ 6e16f844-9365-43af-9ea7-2984808f1fd5
X = data[:, Not(:survival)];

# ╔═╡ b6957225-1889-49fb-93e2-f022ca7c3b23
y = data.survival;

# ╔═╡ c2650040-f398-4a2e-bfe0-ce139c6ca879
# ╠═╡ show_logs = false
let
	model = StableRulesClassifier(; max_depth=2, max_rules=8, rng=_rng())
	mach = machine(model, X, y)
	fit!(mach)
	mach.fitresult
end

# ╔═╡ 172d3263-2e39-483c-9d82-8c22059e63c3
nodes = sort(data.age);

# ╔═╡ cf1816e5-4e8d-4e60-812f-bd6ae7011d6c
# hideall
ln = length(nodes);

# ╔═╡ de90efc9-2171-4406-93a1-9a213ab32259
# hideall
let
	fig = Figure(; resolution=(800, 100))
	ax = Axis(fig[1, 1])
	scatter!(ax, nodes, fill(1, ln))
	hideydecorations!(ax)
	fig
end

# ╔═╡ 8d1b30bd-0ad2-416e-a36a-f263ef781289
# hideall
index = length(nodes) - 3;

# ╔═╡ 2c1adef4-822e-4dc0-946b-dc574e50b305
# hideall
let
	fig = Figure(; resolution=(800, 100))
	ax = Axis(fig[1, 1])
	scatter!(ax, nodes, fill(1, ln))
	vlines!(ax, [nodes[index]]; color=:red)
	annotation = string(round(nodes[index]; digits=2))
	text!(ax, nodes[index] + 0.003, 1.08; text=annotation, fontsize=11)
	hideydecorations!(ax)
	ylims!(ax, 0.9, 1.2)
	fig
end

# ╔═╡ bfcb5e17-8937-4448-b090-2782818c6b6c
# hideall
subset = collect(S._rand_subset(_rng(3), nodes, round(Int, 0.7 * ln)));

# ╔═╡ dff9eb71-a853-4186-8245-a64206379b6f
# hideall
ls = length(subset);

# ╔═╡ 8fdc24d9-1f6b-4094-9722-6b5b6c713f12
# hideall
_plot_cutpoints(subset)

# ╔═╡ 25ad7a18-f989-40f7-8ef1-4ca506446478
# hideall
let
	fig = Figure(; resolution=(800, 100))
	ax = Axis(fig[1, 1])
	scatter!(ax, subset, fill(1, ls))
	vlines!(ax, [nodes[index]]; color=:red, linestyle=:dash)
	annotation = string(round(nodes[index]; digits=2))
	text!(ax, nodes[index] + 0.003, 1.08; text=annotation, fontsize=11)
	hideydecorations!(ax)
	ylims!(ax, 0.9, 1.2)
	fig
end

# ╔═╡ 4935d8f5-32e1-429c-a8c1-84c242eff4bf
# hideall
_plot_cutpoints(nodes)

# ╔═╡ a64dae3c-3b97-4076-98f4-3c9a0e5c0621
# hideall
function _odds_plot(e::PerformanceEvaluation)
	w, h = (1000, 300)
	fig = Figure(; resolution=(w, h))
	grid = fig[1, 1:2] = GridLayout()

	fitresults = getproperty.(e.fitted_params_per_fold, :fitresult)
	feature_names = String[]
	for fitresult in fitresults
		for rule in fitresult.rules
			name = only(rule.path.splits).splitpoint.feature_name
			push!(feature_names, name)
		end
	end

	names = sort(unique(feature_names))
	subtitle = "Ratio"
	
	max_height = maximum(maximum.(getproperty.(fitresults, :weights)))

	importances = _sum_weights.(Ref(fitresults), names)

	matching_rules = DataFrame(; names, importance=importances)
	sort!(matching_rules, :importance; rev=true)
	names = matching_rules.names
	l = length(names)
	
	for (i, feature_name) in enumerate(names)
		yticks = (1:1, [feature_name])
		ax = i == l ? 
			Axis(grid[i, 1:3]; yticks, xlabel="Ratio") : 
			Axis(grid[i, 1:3]; yticks)
		vlines!(ax, [0]; color=:gray, linestyle=:dash)
		xlims!(ax, -1, 1)
		ylabel = feature_name

		name = feature_name

		rules_weights = map(fitresults) do fitresult
			index = _rule_index(fitresult, feature_name)
			isnothing(index) && return nothing
			rule = fitresult.rules[index]::SIRUS.Rule
			return (rule, fitresult.weights[index])
		end
		rw::Vector{Tuple{SIRUS.Rule,Float64}} = 
			filter(!isnothing, rules_weights)
		thresholds = _threshold.(first.(rw))
		t_mean = round(mean(thresholds); digits=1)
		t_std = round(std(thresholds); digits=1)
		
		for (rule, weight) in rw
			left = last(rule.then)::Float64
			right = last(rule.otherwise)::Float64
			t::Float64 = _threshold(rule)
			ratio = log((right) / (left))
			# area = πr²
			markersize = 50 * sqrt(weight / π)
			scatter!(ax, [ratio], [1]; color=:black, markersize)
		end
		hideydecorations!(ax; ticklabels=false)

		axr = i == l ?
			Axis(grid[i, 4:5]; xlabel="Location") :
			Axis(grid[i, 4:5])
		D = data[:, feature_name]
		hist!(axr, D; scale_to=1)
		vlines!(axr, thresholds; color=:black, linestyle=:dash)

		if i < l
			hidexdecorations!(ax)
		else
			hidexdecorations!(ax; ticks=false, ticklabels=false)
		end
	
		hideydecorations!(axr)
		hidexdecorations!(axr; ticks=false, ticklabels=false)
	end

	rowgap!(grid, 5)
	colgap!(grid, 50)
	return fig
end;

# ╔═╡ 4dcd564a-5b2f-4eae-87d6-c2973b828282
_filter_rng(hyper::NamedTuple) = Base.structdiff(hyper, (; rng=:foo));

# ╔═╡ 7a9a0242-a7ba-4508-82fd-a48084525afe
_pretty_name(modeltype) = last(split(string(modeltype), '.'));

# ╔═╡ 6a539bb4-f51f-4efa-af48-c43318ed2502
_hyper2str(hyper::NamedTuple) = hyper == (;) ? "(;)" : string(hyper)::String;

# ╔═╡ 1d08ca81-a18a-4a74-992c-14243d2ea7dc
function _score(e::PerformanceEvaluation)
	return round(only(e.measurement); digits=2)
end;

# ╔═╡ cece10be-736e-4ee1-8c57-89beb0608a92
function _evaluate(modeltype, hyperparameters, X, y)
    model = modeltype(; hyperparameters...)
	e = _evaluate(model, X, y)
	row = (;
	    Model=_pretty_name(modeltype),
	    Hyperparameters=_hyper2str(_filter_rng(hyperparameters)),
	    AUC=_score(e),
	    se=round(only(MLJ.MLJBase._standard_errors(e)); digits=2)
	)
	(; e, row)
end;

# ╔═╡ 9e313f2c-08d9-424f-9ea4-4a4641371360
tree_evaluations = let
	model = DecisionTreeClassifier(; max_depth=2, rng=_rng())
	_evaluate(model, X, y)
end;

# ╔═╡ 39fd9deb-2a27-4c28-ae06-2a36c4c54427
let
	tree = tree_evaluations.fitted_params_per_fold[1].raw_tree
	_io2text() do io
		DecisionTree.print_tree(io, tree; feature_names=names(data))
	end
end

# ╔═╡ 368b6fc1-1cf1-47b5-a746-62c5786dc143
let
	tree = tree_evaluations.fitted_params_per_fold[2].raw_tree
	_io2text() do io
		DecisionTree.print_tree(io, tree; feature_names=names(data))
	end
end

# ╔═╡ ab103b4e-24eb-4575-8c04-ae3fd9ec1673
# ╠═╡ show_logs = false
e1 = let
	model = DecisionTreeClassifier
	hyperparameters = (; max_depth=2, rng=_rng())
	_evaluate(model, hyperparameters, X, y)
end;

# ╔═╡ 6ea43d21-1cc0-4bca-8683-dce67f592949
# ╠═╡ show_logs = false
e2 = let
	model = StableRulesClassifier
	hyperparameters = (; max_depth=2, max_rules=8, rng=_rng())
	_evaluate(model, hyperparameters, X, y)
end

# ╔═╡ 88a708a7-87e8-4f97-b199-70d25ba91894
# ╠═╡ show_logs = false
e3 = let
	model = StableRulesClassifier
	hyperparameters = (; max_depth=2, max_rules=25, rng=_rng())
	_evaluate(model, hyperparameters, X, y)
end;

# ╔═╡ 86ed4d56-23e6-4b4d-9b55-7067124da27f
e4 = let
	model = StableRulesClassifier
	hyperparameters = (; max_depth=1, max_rules=25, rng=_rng())
	_evaluate(model, hyperparameters, X, y)
end;

# ╔═╡ 923affb5-b4ca-4b50-baa5-af29204d2081
# hideall
_odds_plot(e4.e)

# ╔═╡ 7fad8dd5-c0a9-4c45-9663-d40a464bca77
# hideall
fitresults = getproperty.(e4.e.fitted_params_per_fold, :fitresult);

# ╔═╡ 5d875f9d-a0aa-47b0-8a75-75bb280fa1ba
# ╠═╡ show_logs = false
e5 = let
	model = StableForestClassifier
	hyperparameters = (; max_depth=2, rng=_rng())
	_evaluate(model, hyperparameters, X, y)
end;

# ╔═╡ 263ea81f-5fd6-4414-a571-defb1cabab4b
# ╠═╡ show_logs = false
e6 = let
	model = LGBMClassifier
	hyperparameters = (;)
	_evaluate(model, hyperparameters, X, y)
end;

# ╔═╡ 78ba7c69-10df-49d8-8fda-674a1ab05593
e7 = let
	model = LGBMClassifier
	hyperparameters = (; max_depth=2)
	_evaluate(model, hyperparameters, X, y)
end;

# ╔═╡ 622beb62-51ac-4b44-9409-550e5f422fe4
#hideall
results = let
	df = DataFrame(getproperty.([e6, e7, e1, e5, e3, e2, e4], :row))
	df[!, :Interpretability] = ["Medium", "Medium", "High", "Low", "High", "High", "High"]
	df[!, :Stability] = ["High", "High", "Low", "High", "High", "High", "High"]
	df[!, :AUC] = map(df.AUC) do score
		text = string(score)
		length(text) < 4 ? text * '0' : text
	end
	rename!(df, :se => "1.96*SE")
end

# ╔═╡ Cell order:
# ╠═7c10c275-54d8-4f1a-947f-7861199cdf21
# ╠═e9028115-d098-4c61-a82f-d4553fe654f8
# ╠═b1c17349-fd80-43f1-bbc2-53fdb539d1c0
# ╠═348d1235-87f2-4e8f-8f42-be89fef5bf87
# ╠═961aa273-d97b-497f-a79a-06bf89dc34b0
# ╠═6e16f844-9365-43af-9ea7-2984808f1fd5
# ╠═b6957225-1889-49fb-93e2-f022ca7c3b23
# ╠═4c8dd68d-b193-4846-8d93-ab33512c3fa2
# ╠═f75aa57f-6e84-4f7e-88e4-11a00cb9ad2b
# ╠═9e313f2c-08d9-424f-9ea4-4a4641371360
# ╠═e5a45b1a-d761-4279-834b-216df2a1dbb5
# ╠═39fd9deb-2a27-4c28-ae06-2a36c4c54427
# ╠═d38f8814-c7b8-4911-9c63-d99b646b4486
# ╠═368b6fc1-1cf1-47b5-a746-62c5786dc143
# ╠═5318414e-5c87-4be0-bcd0-b6efd4eee5b9
# ╠═d816683b-2f7d-45a7-bd40-42f554a48b1b
# ╠═4b67c47a-ee98-495e-bb1b-41db83c11cd4
# ╠═172d3263-2e39-483c-9d82-8c22059e63c3
# ╠═cf1816e5-4e8d-4e60-812f-bd6ae7011d6c
# ╠═de90efc9-2171-4406-93a1-9a213ab32259
# ╠═0d121fa3-fbfa-44e5-904b-64a1622ec91b
# ╠═8d1b30bd-0ad2-416e-a36a-f263ef781289
# ╠═2c1adef4-822e-4dc0-946b-dc574e50b305
# ╠═896e00dc-2ce9-4a9f-acc1-519aec21dd83
# ╠═bfcb5e17-8937-4448-b090-2782818c6b6c
# ╠═dff9eb71-a853-4186-8245-a64206379b6f
# ╠═25ad7a18-f989-40f7-8ef1-4ca506446478
# ╠═ee12350a-627b-4a11-99cb-38c496977d18
# ╠═4935d8f5-32e1-429c-a8c1-84c242eff4bf
# ╠═0cc970cd-b7ed-4782-a520-ff0a76fe0453
# ╠═8fdc24d9-1f6b-4094-9722-6b5b6c713f12
# ╠═01b08d44-4b9b-42e2-bb20-f34cb9b407f3
# ╠═7e1d46b4-5f93-478d-9105-a5b0db1eaf08
# ╠═ab103b4e-24eb-4575-8c04-ae3fd9ec1673
# ╠═6ea43d21-1cc0-4bca-8683-dce67f592949
# ╠═88a708a7-87e8-4f97-b199-70d25ba91894
# ╠═86ed4d56-23e6-4b4d-9b55-7067124da27f
# ╠═5d875f9d-a0aa-47b0-8a75-75bb280fa1ba
# ╠═263ea81f-5fd6-4414-a571-defb1cabab4b
# ╠═78ba7c69-10df-49d8-8fda-674a1ab05593
# ╠═622beb62-51ac-4b44-9409-550e5f422fe4
# ╠═4a4ab7ef-659e-4048-ab16-94ad4cb4328a
# ╠═16de5518-2a16-40ef-87a5-d2acd514d294
# ╠═c2650040-f398-4a2e-bfe0-ce139c6ca879
# ╠═6d0b29b6-61fb-4d16-9389-071892a3d9db
# ╠═3c415a26-803e-4f35-866f-2e582c6c1c45
# ╠═aa93a6c4-d5a0-4c73-9db9-e26c3c3f526b
# ╠═a64dae3c-3b97-4076-98f4-3c9a0e5c0621
# ╠═923affb5-b4ca-4b50-baa5-af29204d2081
# ╠═ab5423cd-c8a9-488e-9bb0-bb41e583c2fa
# ╠═f2fee9a8-7f6f-4213-9046-2f1a8f14a7e6
# ╠═e6b880e9-e263-4818-81e9-bb4105e5c2c1
# ╠═7fad8dd5-c0a9-4c45-9663-d40a464bca77
# ╠═cfd908a0-1ee9-461d-9309-d4ffe738ba8e
# ╠═e7f396dc-38a7-40f7-9e5b-6fbea9d61789
# ╠═7c688412-d1b4-492d-bda2-0b9181057d4d
# ╠═e1890517-7a44-4814-999d-6af27e2a136a
# ╠═f833dab6-31d4-4353-a68b-ef0501d606d4
# ╠═ede038b3-d92e-4208-b8ab-984f3ca1810e
# ╠═93a7dd3b-7810-4021-bf6e-ae9c04acea46
# ╠═be324728-1b60-4584-b8ea-c4fe9e3466af
# ╠═7ad3cf67-2acd-44c6-aa91-7d5ae809dfbc
# ╠═0ca8bb9a-aac1-41a7-b43d-314a4029c205
# ╠═9db18ac7-4508-4861-8854-3e19d5218309
# ╠═564598d7-5bdf-4e42-b812-8aca20fa20d4
# ╠═4dcd564a-5b2f-4eae-87d6-c2973b828282
# ╠═7a9a0242-a7ba-4508-82fd-a48084525afe
# ╠═cece10be-736e-4ee1-8c57-89beb0608a92
# ╠═6a539bb4-f51f-4efa-af48-c43318ed2502
# ╠═1d08ca81-a18a-4a74-992c-14243d2ea7dc
