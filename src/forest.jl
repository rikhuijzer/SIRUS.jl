"Supertype for random forests modes: classification or regression."
abstract type Algorithm end

"Supertype for leafs: classification or regression."
abstract type Leaf end

"""
Type which holds the values inside a leaf.
For classification, this is a vector of probabilities of each class.
For regression, this is a vector of one element.

In some sense, regression can be thought of as a special case of
classification, namely as classification with only one class.

!!! note
    Vectors of one element are not as performant as scalars, but the
    alternative here is to have two different types of leafs, which
    results in different types of trees also, which basically
    requires most functions then to become parametric; especially
    in `src/rules.jl`.
"""
const LeafContent = Vector{Float64}

"Return the number of elements in `V` being equal to `x`."
function _count_equal(V::AbstractVector, x)::Int
    c = 0
    @inbounds @simd for v in V
        if x == v
            c += 1
        end
    end
    return c
end

"""
    SplitPoint(feature::Int, value::Float32, feature_name::String)

A location where the tree splits.

Arguments:
- `feature`: Feature index.
- `value`: Value of split.
- `feature_name`: Name of the feature which is used for pretty printing.
"""
struct SplitPoint
    feature::Int
    value::Float32
    feature_name::String255

    function SplitPoint(feature::Int, value::Float32, feature_name::String)
        return new(feature, value, String255(feature_name))
    end
end

_feature(sp::SplitPoint) = sp.feature
_value(sp::SplitPoint) = sp.value
_feature_name(sp::SplitPoint) = sp.feature_name

"Return a random subset of `V` sampled without replacement."
function _rand_subset(rng::AbstractRNG, V::AbstractVector, n::Int)
    return view(shuffle(rng, V), 1:n)
end

"""
Return a view on all `y` for which the `comparison` holds in `data`.

The mutable `y_view` is used to have a view of `y` in continuous memory.
"""
function _view_y!(y_view, data, y, comparison, cutpoint)
    len = 0
    @inbounds for i in eachindex(data)
        value = @inbounds data[i]
        result = comparison(value, cutpoint)
        if result
            len += 1
            @inbounds y_view[len] = y[i]
        end
    end
    return @inbounds view(y_view, 1:len)
end

function _max_split_candidates(X)::Int
    return round(Int, sqrt(nfeatures(X)))
end

"""
Return the split for which the gini index is maximized.
This function receives the cutpoints for the whole dataset `D` because `X` can be a subset of `D`.
For a walkthrough of the CART algorithm, see https://youtu.be/LDRbO9a6XPU.
"""
function _split(
        rng,
        algo::Algorithm,
        X,
        y::AbstractVector,
        classes::AbstractVector,
        colnms::Vector{String},
        cps::Vector{Cutpoints};
        max_split_candidates::Int=_max_split_candidates(X)
    )::Union{Nothing, SplitPoint}
    score_improved::Bool = false
    best_score = _start_score(algo)
    best_score_feature = 0
    best_score_cutpoint = 0.0

    p = nfeatures(X)
    mc = max_split_candidates
    possible_features = mc == p ? (1:p) : _rand_subset(rng, 1:p, mc)
    reused_data = _reused_data(algo, y, classes)

    yl = Vector{eltype(y)}(undef, length(y))
    yr = Vector{eltype(y)}(undef, length(y))
    feat_data = Vector{eltype(X)}(undef, length(y))
    for feature in possible_features
        @inbounds for i in eachindex(feat_data)
            feat_data[i] = @inbounds X[i, feature]
        end
        for cutpoint in cps[feature]
            vl = _view_y!(yl, feat_data, y, <, cutpoint)
            isempty(vl) && continue
            vr = _view_y!(yr, feat_data, y, ≥, cutpoint)
            isempty(vr) && continue
            current_score = _current_score(algo, y, vl, vr, classes, reused_data)
            if _score_improved(algo, best_score, current_score)
                score_improved = true
                best_score = current_score
                best_score_feature = feature
                best_score_cutpoint = cutpoint
            end
        end
    end
    if score_improved
        feature_name = colnms[best_score_feature]
        return SplitPoint(best_score_feature, best_score_cutpoint, feature_name)
    else
        return nothing
    end
end

struct Node
    splitpoint::SplitPoint
    left::Union{Node, Leaf}
    right::Union{Node, Leaf}
end

children(node::Node) = [node.left, node.right]
nodevalue(node::Node) = node.splitpoint

"""
Return a view on all rows in `X` and `y` for which the `comparison` holds in `X[:, feature]`.
"""
function _view_X_y!(mask, X, y, splitpoint::SplitPoint, comparison)
    data = @inbounds view(X, :, splitpoint.feature)
    @assert length(data) == length(y)
    len = 0
    for i in eachindex(y)
        value = @inbounds data[i]
        result = comparison(value, splitpoint.value)
        @inbounds mask[i] = result
    end
    mask_subset = view(mask, 1:length(y))
    X_view = @inbounds view(X, mask_subset, :)
    y_view = @inbounds view(y, mask_subset)
    return (X_view, y_view)
end

function _verify_lengths(X, y)
    l1 = size(X, 1)
    l2 = length(y)
    if l1 != l2
        error("Expected X and y to have the same number of rows, but got $l1 and $l2 rows")
    end
end

"Return the root node of a stable decision tree fitted on `X` and `y`."
function _tree!(
        rng::AbstractRNG,
        algo::Algorithm,
        mask::Vector{Bool},
        X,
        y::AbstractVector,
        classes::AbstractVector,
        colnms::Vector{String}=colnames(X);
        max_split_candidates=_max_split_candidates(X),
        depth=0,
        max_depth=2,
        q=10,
        cps::Vector{Cutpoints}=cutpoints(X, q),
        min_data_in_leaf=5
    )
    if X isa Tables.MatrixTable
        error("Not implemented for arbitrary tables yet. Pass a matrix instead")
    end
    _verify_lengths(X, y)
    if depth == max_depth
        return Leaf(algo, classes, y)
    end
    sp = _split(rng, algo, X, y, classes, colnms, cps; max_split_candidates)
    if isnothing(sp) || length(y) ≤ min_data_in_leaf
        return Leaf(algo, classes, y)
    end
    depth += 1

    left = let
        _X, yl = _view_X_y!(mask, X, y, sp, <)
        _tree!(rng, algo, mask, _X, yl, classes, colnms; cps, depth, max_depth)
    end
    right = let
        _X, yr = _view_X_y!(mask, X, y, sp, ≥)
        _tree!(rng, algo, mask, _X, yr, classes, colnms; cps, depth, max_depth)
    end
    node = Node(sp, left, right)
    return node
end

"""
Predict `y` for a data vector defined by `x`.
Also pass a vector if the data has only one feature.
"""
function _predict(node::Node, x::AbstractVector)
    feature = node.splitpoint.feature
    value = node.splitpoint.value
    if x[feature] < value
        return _predict(node.left, x)
    else
        return _predict(node.right, x)
    end
end
function _predict(node::Node, x::Union{Tables.MatrixRow, Tables.ColumnsRow})
    return _predict(node, collect(x))
end
function _predict(node::Node, X::AbstractMatrix)
    return _predict.(Ref(node), eachrow(X))
end

abstract type StableModel end

"""
    StableForest{T} <: StableModel

Stable forest containing the trees, algorithm, and classes.
This object is created by the MLJ-interface.
"""
struct StableForest{T} <: StableModel
    trees::Vector{Union{Node,Leaf}}
    algo::Algorithm
    classes::Vector{T}
end
_elements(model::StableForest) = model.trees

"Increase the state of `rng` by `i`."
_change_rng_state!(rng::AbstractRNG, i::Int) = seed!(rng, i)

"""
Return an unique and sorted vector of classes based on `y`.
The vector is sorted to ensure that class ordering is the same between cross-validations.
This holds as long as each class is in each fold.
"""
_classes(y::AbstractVector) = sort(unique(y))

const PARTIAL_SAMPLING_DEFAULT = 0.7
const N_TREES_DEFAULT = 1_000
const MAX_DEPTH_DEFAULT = 2

"""
Return a random forest.

Arguments:

- `partial_sampling::Real=$PARTIAL_SAMPLING_DEFAULT`:
    Proportion of samples to use per tree fit.
- `n_trees=$N_TREES_DEFAULT`: Number of trees to fit.
- `max_depth=$MAX_DEPTH_DEFAULT`: Max depth of each tree.

!!! note
    Each tree in a random forest is built by taking a random sample from the dataset (bootstrapped sample).
    And unlike in bagging, each tree also gets to see only a set of `m` randomly chosen features,
    where for some total number of features `p`, then `m = sqrt(p)` (James et al., [2014](https://doi.org/10.1007/978-1-0716-1418-1)).
"""
function _forest(
        rng::AbstractRNG,
        algo::Algorithm,
        X::AbstractMatrix,
        y::AbstractVector,
        colnms::Vector{String};
        partial_sampling::Real=PARTIAL_SAMPLING_DEFAULT,
        n_trees::Int=N_TREES_DEFAULT,
        max_depth::Int=MAX_DEPTH_DEFAULT,
        q::Int=10,
        min_data_in_leaf::Int=5
    )
    if 2 < max_depth
        error("""
              Tree depth is too high. Rule filtering for a depth above 2 is not implemented.
              In the original paper, the authors also advice to use a depth of no more than 2.
              """)
    end
    if max_depth < 1
        error("Minimum tree depth is 1; got $max_depth")
    end
    # It is essential for the stability to determine the cutpoints over the whole dataset.
    cps = cutpoints(X, q)
    classes = algo isa Classification ? _classes(y) : []

    max_split_candidates = _max_split_candidates(X)
    n_samples = floor(Int, partial_sampling * length(y))

    trees = Vector{Union{Node,Leaf}}(undef, n_trees)
    # Threads.@threads
    for i in 1:n_trees
        _rng = copy(rng)
        _change_rng_state!(_rng, i)
        # Note that this is sampling with replacement; keep it this way.
        rows = rand(_rng, 1:length(y), n_samples)
        _X = X[rows, :]
        _y = view(y, rows)
        mask = Vector{Bool}(undef, length(y))
        tree = _tree!(
            _rng,
            algo,
            mask,
            _X,
            _y,
            classes,
            colnms;
            max_split_candidates,
            max_depth,
            q,
            cps,
            min_data_in_leaf
        )
        trees[i] = tree
    end
    return StableForest(trees, algo, classes)
end

function _forest(rng::AbstractRNG, algo::Algorithm, X, y; kwargs...)
    if !(X isa AbstractMatrix || Tables.istable(X))
        error("Input `X` doesn't satisfy the Tables.jl interface.")
    end
    # Tables doesn't assume the data fits in memory so that complicates things a lot.
    # Implementing out-of-memory trees is a problem for later.
    return _forest(rng, algo, matrix(X), y, colnames(X); kwargs...)
end

function _isempty_error(::StableForest)
    throw(AssertionError("The forest contains no trees"))
end

function _apply_statistic(V::AbstractVector{<:AbstractVector}, f::Function)
    M = reduce(hcat, V)
    return [round(f(row); sigdigits=3) for row in eachrow(M)]
end

_mean(V::AbstractVector{<:AbstractVector}) = _apply_statistic(V, mean)
_median(V::AbstractVector{<:AbstractVector}) = _apply_statistic(V, median)
_mode(V::AbstractVector{<:AbstractVector}) = _apply_statistic(V, mode)

function _predict(forest::StableForest, row::AbstractVector)
    isempty(_elements(forest)) && _isempty_error(forest)
    predictions = [_predict(tree, row) for tree in forest.trees]
    if forest.algo isa Classification
        return _median(predictions)
    else
        m = median(predictions)
        @assert m isa Number
        return m
    end
end

function _predict(model::StableModel, X::AbstractMatrix)
    predictions = [_predict(model, row) for row in eachrow(X)]
    if model.algo isa Classification
        P = reduce(hcat, predictions)'
        return UnivariateFinite(model.classes, P; pool=missing)
    else
        return only.(predictions)
    end
end

function _predict(model::StableModel, X)
    if !Tables.istable(X)
        error("Expected a Table but got $(typeof(X))")
    end
    return _predict(model, Tables.matrix(X))
end
