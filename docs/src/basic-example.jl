### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ 8ca5e9f4-539c-11ee-0b5a-ab77d2a5bfbf
# ╠═╡ show_logs = false
begin
	ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

	PKGDIR = dirname(dirname(@__DIR__))
    DOCS_DIR = dirname(@__DIR__)
	using Pkg: Pkg
	Pkg.activate(DOCS_DIR)
	Pkg.develop(; path=PKGDIR)
end

# ╔═╡ 1c1bd75a-9266-4256-bfea-ad60dd1c1d1c
begin
	using CategoricalArrays: categorical
	using CSV: CSV
	using DataDeps: DataDeps, DataDep, @datadep_str
	using DataFrames
	using MLJ
	using PlutoUI: TableOfContents # hide
	using StableRNGs: StableRNG
	using SIRUS: StableRulesClassifier
end

# ╔═╡ 3071aa12-92df-47b2-a5ce-a54a7110ab6a
md"""
# Basic Example

This page shows a basic example for using SIRUS.jl on a dataset via the Machine Learning Julia (MLJ.jl) interface.
For more details on what SIRUS is and how it works, see the
[Advanced Example](/dev/binary-classification).
"""

# ╔═╡ 3e9f7866-edaa-460a-81d4-abf76ab066dc
# hideall
TableOfContents()

# ╔═╡ 0b4f89c4-ccb0-4cc9-b7bb-3f630ba2398c
md"""
To show the algorithm, we'll use Haberman's survival dataset.
We load it via `DataDeps.jl` so that we can use a checksum for verification and to cache the dataset.
"""

# ╔═╡ 75550619-b310-4c66-9371-93656f78765c
# ╠═╡ show_logs = false
let
    name = "Haberman"
    message = "Haberman's Survival Data Set"
    remote_path = "https://github.com/rikhuijzer/haberman-survival-dataset/releases/download/v1.0.0/haberman.csv"
    checksum = "a7e9aeb249e11ac17c2b8ea4fdafd5c9392219d27cb819ffaeb8a869eb727a0f"
    DataDeps.register(DataDep(name, message, remote_path, checksum))
end;

# ╔═╡ 05aaa007-0fe0-44ef-b815-ecf9e5f474f7
md"""
After dataset registration, we can load it into a `DataFrame`:
"""

# ╔═╡ ac2d7dbc-364f-437f-b66f-8eb288395275
data = let
    dir = datadep"Haberman"
    path = joinpath(dir, "haberman.csv")
    df = CSV.read(path, DataFrame)
    df[!, :survival] = categorical(df.survival)
    df
end

# ╔═╡ 08a4ca2b-bc65-4c29-9528-f4789272143a
md"And split it into features (`X`) and outcomes (`y`):"

# ╔═╡ e037d952-e489-41b6-afc9-317a8c17e6c4
X = select(data, Not(:survival));

# ╔═╡ 2f921f63-5148-4726-9839-c84217f60e0b
y = data.survival;

# ╔═╡ a1764625-4b7a-42f3-9e61-3d26122d86da
md"""
Next, we can load the model that we want to use.
Since Haberman's outcome column (`survival`) contains 0's and 1's, we use the `StableRulesClassifier`.
We can make the `StableRulesClassifier` symbol available via MLJ's `@load`:
```julia
StableRulesClassifier = @load StableRulesClassifier pkg="SIRUS"
```
or directly via
```julia
using SIRUS: StableRulesClassifier
```
"""

# ╔═╡ ccce5f3e-e396-4765-bf5f-6f79e905aca8
model = StableRulesClassifier(; rng=StableRNG(1), q=4, max_depth=2, max_rules=8);

# ╔═╡ 97c9ea2a-2897-472b-b15e-215f40049cf5
md"""
Next, we will show two common use-cases:

1. fit the model to the full dataset and
2. fit the model to cross-validation folds (to evaluate model performance).
"""

# ╔═╡ b1281f29-61d7-4c43-960c-e516464ea213
md"""
## Fitting to the full dataset
"""

# ╔═╡ c77e3efb-9170-4675-b053-b99cdb8db853
# ╠═╡ show_logs = false
mach = let
	mach = machine(model, X, y)
	MLJ.fit!(mach)
end;

# ╔═╡ d90d86de-943e-4cad-a5bd-fecac2681b98
md"And inspect the fitted model:"

# ╔═╡ 0c9b2e27-beb0-4f12-b8fd-33dc67f598c1
mach.fitresult

# ╔═╡ 89b12064-5d46-436c-b697-0a4dc527d586
md"""
This shows that the model contains $(length(mach.fitresult.rules)) rules where the first rule, for example, can be interpreted as

_If the number of detected axillary nodes is lower than 7, then take 0.238 and otherwise take 0.046._

This calculation is done for all $(length(mach.fitresult.rules)) rules and the score is summed to get a prediction.
"""

# ╔═╡ 13ad02dd-c557-4599-8502-f85d20234ed0
md"""
The predictions are of the type `UnivariateFinite` from MLJ's `CategoricalDistributions.jl`:
"""

# ╔═╡ e732756b-7aaa-4fcc-b90f-1b418208c5af
predictions = predict(mach, X)

# ╔═╡ 58711147-9f89-465a-9e21-ab1d64e03c2d
md"""
To get the underlying predictions out of these objects, use `pdf`.
For example, to get the prediction for the class 0 for the first datapoint, use:
"""

# ╔═╡ ed969c5c-6f58-4b8c-825b-fcf04da74036
pdf(predictions[1], 0)

# ╔═╡ 1ca8a8b1-0623-47d7-8900-41056e0b21ee
md"""
See <https://alan-turing-institute.github.io/MLJ.jl/dev/getting_started/#Fit-and-predict> for more information.
"""

# ╔═╡ ece3f092-368e-41af-994a-e814f2267f48
md"""
## Model Evaluation via Cross-Validation

Let's define our Cross-Validation (CV) strategy with 10 folds.
Also, we enable shuffling to make it more likely that our model sees cases from both `survival` classes:
"""

# ╔═╡ dfc6f708-3d26-4102-92c6-33cee32e438c
resampling = CV(; rng=StableRNG(1), nfolds=10, shuffle=true);

# ╔═╡ d2905680-552d-4a9a-b3f1-7dd27cbf703f
md"""
We use the Area Under the Curve (AUC) measure since that measure is appropriate for binary classification tasks.
More specifically, the measure gives the area under the receiver operating characteristic curve.
For this measure, a score of 0.5 means that our model is as good (or bad, actually) as random guessing, and a score of 0.0 means predicting all cases wrong and 1.0 means predicting all cases correctly.
"""

# ╔═╡ b8c6c9e0-679e-41d5-80c0-ffd65e652489
# ╠═╡ show_logs = false
evaluate(model, X, y; resampling, measure=auc)

# ╔═╡ Cell order:
# ╠═3071aa12-92df-47b2-a5ce-a54a7110ab6a
# ╠═8ca5e9f4-539c-11ee-0b5a-ab77d2a5bfbf
# ╠═3e9f7866-edaa-460a-81d4-abf76ab066dc
# ╠═1c1bd75a-9266-4256-bfea-ad60dd1c1d1c
# ╠═0b4f89c4-ccb0-4cc9-b7bb-3f630ba2398c
# ╠═75550619-b310-4c66-9371-93656f78765c
# ╠═05aaa007-0fe0-44ef-b815-ecf9e5f474f7
# ╠═ac2d7dbc-364f-437f-b66f-8eb288395275
# ╠═08a4ca2b-bc65-4c29-9528-f4789272143a
# ╠═e037d952-e489-41b6-afc9-317a8c17e6c4
# ╠═2f921f63-5148-4726-9839-c84217f60e0b
# ╠═a1764625-4b7a-42f3-9e61-3d26122d86da
# ╠═ccce5f3e-e396-4765-bf5f-6f79e905aca8
# ╠═97c9ea2a-2897-472b-b15e-215f40049cf5
# ╠═b1281f29-61d7-4c43-960c-e516464ea213
# ╠═c77e3efb-9170-4675-b053-b99cdb8db853
# ╠═d90d86de-943e-4cad-a5bd-fecac2681b98
# ╠═0c9b2e27-beb0-4f12-b8fd-33dc67f598c1
# ╠═89b12064-5d46-436c-b697-0a4dc527d586
# ╠═13ad02dd-c557-4599-8502-f85d20234ed0
# ╠═e732756b-7aaa-4fcc-b90f-1b418208c5af
# ╠═58711147-9f89-465a-9e21-ab1d64e03c2d
# ╠═ed969c5c-6f58-4b8c-825b-fcf04da74036
# ╠═1ca8a8b1-0623-47d7-8900-41056e0b21ee
# ╠═ece3f092-368e-41af-994a-e814f2267f48
# ╠═dfc6f708-3d26-4102-92c6-33cee32e438c
# ╠═d2905680-552d-4a9a-b3f1-7dd27cbf703f
# ╠═b8c6c9e0-679e-41d5-80c0-ffd65e652489
