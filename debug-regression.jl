### A Pluto.jl notebook ###
# v0.19.32

using Markdown
using InteractiveUtils

# ╔═╡ 44f5f78b-d317-4662-ab6d-307f0701a11b
using Revise

# ╔═╡ f67cddf9-7304-47e2-aa2f-48ec93275554
root_dir = @__DIR__

# ╔═╡ 106c3646-8d4a-11ee-0601-27fe3abdc084
# ╠═╡ show_logs = false
x = let
	using Pkg: Pkg
	Pkg.add("TestEnv")
	Pkg.activate(root_dir)
	using TestEnv
	TestEnv.activate()
	Pkg.add("CairoMakie")
	using CairoMakie
end;

# ╔═╡ ff2b378b-75a6-4e0a-b13b-6bc69442180b
let
	x
	using CategoricalArrays:
	    CategoricalValue,
	    CategoricalVector,
	    categorical,
	    unwrap
	using CSV: CSV
	using DataDeps: DataDeps, DataDep, @datadep_str
	using Documenter: DocMeta, doctest
	using MLDatasets:
	    BostonHousing,
	    Iris,
	    Titanic
	using DataFrames:
	    DataFrames,
	    DataFrame,
	    Not,
	    dropmissing!,
	    rename!,
	    rename,
	    select
	using DecisionTree: DecisionTree
	using MLJBase:
	    CV,
	    MLJBase,
	    PerformanceEvaluation,
	    evaluate,
	    mode,
	    fit!,
	    machine,
	    make_blobs,
	    make_moons,
	    make_regression,
	    predict
	using MLJDecisionTreeInterface: DecisionTreeClassifier, DecisionTreeRegressor
	using MLJLinearModels: LogisticClassifier, LinearRegressor, MultinomialClassifier
	using MLJTestInterface: MLJTestInterface
	using MLJXGBoostInterface: XGBoostClassifier, XGBoostRegressor
	using Random: shuffle, seed!
	using StableRNGs: StableRNG
	using StatisticalMeasures:
	    accuracy,
	    auc,
	    rsq
	using SIRUS
	using Statistics: mean, var
	using Tables: Tables
	using Test
end

# ╔═╡ 5e0aa5fc-c78d-4f2c-94dd-c73fbc726a8d
include(joinpath(root_dir, "test/preliminaries.jl"))

# ╔═╡ 14957f30-40c5-4445-8efc-68ea3c8e3e0f
X, y = boston();

# ╔═╡ 24366d12-3df7-4934-b57e-f489e1b76e9a
X

# ╔═╡ 693b7971-878f-40be-9989-1a37f1d8511f
y

# ╔═╡ ee87d4f9-08c6-4d25-ad68-8e69e675d3c7
fr = let
	hyper = (; rng=_rng(), max_depth=2, n_trees=100)
	measure = accuracy
    _evaluate!(results, "iris", StableForestClassifier, hyper; measure)
end

# ╔═╡ ebc2ba6f-a7da-4290-877a-59a40045d021
fr.fitted_params_per_fold[1].fitresult

# ╔═╡ f306ae6f-fa3d-484b-af78-47f012c9ac07
forest = fr.fitted_params_per_fold[2].fitresult

# ╔═╡ 932a9f80-6ec4-41e1-b550-4fca88ecb68a
rr = let
	# Increasing max_rules decreases score, which makes no sense.
	hyper = (; rng=_rng(), max_depth=2, max_rules=30, lambda=100, q=20)
	measure = rsq
    _evaluate!(results, "boston", StableRulesRegressor, hyper; measure)
end

# ╔═╡ c9b5c558-30c2-4b53-a10d-4396878d24b0
only(rr.per_fold)

# ╔═╡ 7dc325e0-07b5-4f65-9d4e-ba98c5e38dc5
rr.fitted_params_per_fold[1].fitresult

# ╔═╡ 4af16b84-683f-4a16-abb5-855a98a14263
rr.fitted_params_per_fold[1].fitresult.weights

# ╔═╡ d2d7d82b-18bb-4b7c-ac77-4b8ceaa1a8e9
forest

# ╔═╡ Cell order:
# ╠═44f5f78b-d317-4662-ab6d-307f0701a11b
# ╠═f67cddf9-7304-47e2-aa2f-48ec93275554
# ╠═106c3646-8d4a-11ee-0601-27fe3abdc084
# ╠═5e0aa5fc-c78d-4f2c-94dd-c73fbc726a8d
# ╠═ff2b378b-75a6-4e0a-b13b-6bc69442180b
# ╠═14957f30-40c5-4445-8efc-68ea3c8e3e0f
# ╠═24366d12-3df7-4934-b57e-f489e1b76e9a
# ╠═693b7971-878f-40be-9989-1a37f1d8511f
# ╠═ee87d4f9-08c6-4d25-ad68-8e69e675d3c7
# ╠═ebc2ba6f-a7da-4290-877a-59a40045d021
# ╠═f306ae6f-fa3d-484b-af78-47f012c9ac07
# ╠═932a9f80-6ec4-41e1-b550-4fca88ecb68a
# ╠═c9b5c558-30c2-4b53-a10d-4396878d24b0
# ╠═7dc325e0-07b5-4f65-9d4e-ba98c5e38dc5
# ╠═4af16b84-683f-4a16-abb5-855a98a14263
# ╠═d2d7d82b-18bb-4b7c-ac77-4b8ceaa1a8e9
