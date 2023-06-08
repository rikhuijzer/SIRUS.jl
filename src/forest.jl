"""
    _gini(y::AbstractVector, classes::AbstractVector)

Return the Gini index for a vector outcomes `y` and `classes`.
Here, `y` is usually a view on the outcome values in some region.
Inside that region, `gini` is a measure of node purity.
If all values in the region have the same class, then gini is zero.
The equation is mentioned on Wikipedia as
``1 - \\sum{class \\in classes} p_i^2``
for ``p_i`` be the fraction (proportion) of items labeled with class ``i`` in the set.
"""
function _gini(y::AbstractVector, classes)
    len_y = length(y)
    impurity = 1.0
    for class in classes
        c = @inbounds count(==(class), y)
        proportion = c / len_y
        impurity -= proportion^2
    end
    return impurity
end

function _information_gain(
        y,
        yl,
        yr,
        classes,
        starting_impurity::Real
    )
    p = length(yl) / length(y)
    impurity_change = p * _gini(yl, classes) + (1 - p) * _gini(yr, classes)
    return starting_impurity - impurity_change
end

"""
Return a rough estimate for the index of the cutpoint.
Choose the highest suitable index if there is more than one suitable index.
The reason is that this will split the data nicely in combination with the `<` used later on.
For example, for [1, 2, 3, 4], both 2 and 3 satisfy the 0.5 quantile.
In this case, we pick the ceil, so 3.
Next, the tree will split on 3, causing left (<) to contain 1 and 2 and right (≥) to contain 3 and 4.
"""
_rough_cutpoint_index_estimate(n::Int, quantile::Real) = Int(ceil(quantile * n))

"Return the empirical `quantile` for data `V`."
function _empirical_quantile(V::AbstractVector, quantile::Real)
    @assert 0.0 ≤ quantile ≤ 1.0
    n = length(V)
    index = _rough_cutpoint_index_estimate(n, quantile)
    if index == 0
        index = 1
    end
    if index == n + 1
        index = n
    end
    sorted = sort(V)
    return Float(sorted[index])
end

"Return a vector of `q` cutpoints taken from the empirical distribution from data `V`."
function _cutpoints(V::AbstractVector, q::Int)
    @assert 2 ≤ q
    # Taking 2 extra to avoid getting minimum(V) and maximum(V) becoming cutpoints.
    # Tree on left and right have always respectively length 0 and 1 then anyway.
    length = q + 2
    quantiles = range(0.0; stop=1.0, length)[2:end-1]
    return Float[_empirical_quantile(V, quantile) for quantile in quantiles]
end

"Return the number of features `p` in a dataset `X`."
_p(X::AbstractMatrix) = size(X, 2)
_p(X) = length(Tables.columnnames(X))

"Set of possible cutpoints, that is, numbers from the empirical quantiles."
const Cutpoints = Vector{Float}

_view_feature(X::AbstractMatrix, feature::Int) = view(X, :, feature)
_view_feature(X, feature::Int) = X[feature]

"""
Return a vector of vectors containing
- one inner vector for each feature in the dataset and
- inner vectors containing the unique cutpoints, that is, `length(V[i])` ≤ `q` for all i in V.

Using unique here to avoid checking splits twice.
"""
function _cutpoints(X, q::Int)
    p = _p(X)
    cutpoints = Vector{Cutpoints}(undef, p)
    for feature in 1:p
        V = _view_feature(X, feature)
        cutpoints[feature] = unique(_cutpoints(V, q))
    end
    return cutpoints
end

"""
    SplitPoint(feature::Int, value::Float, feature_name::String)

A location where the tree splits.

Arguments:
- `feature`: Feature index.
- `value`: Value of split.
- `feature_name`: Name of the feature which is used for pretty printing.
"""
struct SplitPoint
    feature::Int
    value::Float
    feature_name::String255

    function SplitPoint(feature::Int, value::Float, feature_name::String)
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
function _view_y!(y_view, X, feature::Int, y, comparison, cutpoint)
    len = 0
    @inbounds for i in eachindex(y)
        value = @inbounds X[i, feature]
        result = comparison(value, cutpoint)
        if result
            len += 1
            @inbounds y_view[len] = y[i]
        end
    end
    return @inbounds view(y_view, 1:len)
end

"""
Return the split for which the gini index is maximized.
This function receives the cutpoints for the whole dataset `D` because `X` can be a subset of `D`.
For a walkthrough of the CART algorithm, see https://youtu.be/LDRbO9a6XPU.
"""
function _split(
        rng,
        X,
        y::AbstractVector,
        classes::AbstractVector,
        colnames::Vector{String},
        cutpoints::Vector{Cutpoints};
        max_split_candidates::Int=_p(X)
    )
    best_score = 0.0
    best_score_feature = 0
    best_score_cutpoint = 0.0
    p = _p(X)
    mc = max_split_candidates
    possible_features = mc == p ? (1:p) : _rand_subset(rng, 1:p, mc)
    starting_impurity = _gini(y, classes)

    yl = Vector{eltype(y)}(undef, length(y))
    yr = Vector{eltype(y)}(undef, length(y))
    for feature in possible_features
        for cutpoint in cutpoints[feature]
            vl = _view_y!(yl, X, feature, y, <, cutpoint)
            isempty(vl) && continue
            vr = _view_y!(yr, X, feature, y, ≥, cutpoint)
            isempty(vr) && continue
            gain = _information_gain(y, vl, vr, classes, starting_impurity)
            if best_score ≤ gain
                best_score = gain
                best_score_feature = feature
                best_score_cutpoint = cutpoint
            end
        end
    end
    if best_score == 0.0
        return nothing
    end
    feature_name = colnames[best_score_feature]
    return SplitPoint(best_score_feature, best_score_cutpoint, feature_name)
end

const Probabilities = Vector{Float64}

"""
    Leaf

Leaf of a decision tree.
The probabilities are based on the `y`'s falling into the region associated with this leaf.
"""
struct Leaf
    probabilities::Probabilities
end

function Leaf(classes, y)
    l = length(y)
    probabilities = [count(y .== class) / l for class in classes]
    # Not creating a UnivariateFinite because it requires MLJBase
    return Leaf(probabilities)
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

"Return `names(X)` if defined for `X` and string numbers otherwise."
function _colnames(X)::Vector{String}
    fallback() = string.(1:_p(X))
    try
        names = collect(string.(Tables.columnnames(X)))
        if isempty(names)
            return fallback()
        else
            return names
        end
    catch
        return fallback()
    end
end

"""
Return the root node of a stable decision tree fitted on `X` and `y`.

Arguments:
- `max_split_candidates`:
    During random forest creation, the number of split candidates is limited to make the trees less correlated.
    See Section 8.2.2 of https://doi.org/10.1007/978-1-0716-1418-1 for details.
"""
function _tree!(
        rng::AbstractRNG,
        mask::Vector{Bool},
        X,
        y::AbstractVector,
        classes::AbstractVector,
        colnames::Vector{String}=_colnames(X);
        max_split_candidates=_p(X),
        depth=0,
        max_depth=2,
        q=10,
        cutpoints::Vector{Cutpoints}=_cutpoints(X, q),
        min_data_in_leaf=5
    )
    if X isa Tables.MatrixTable
        error("Not implemented for arbitrary tables yet. Pass a matrix instead")
    end
    _verify_lengths(X, y)
    if depth == max_depth
        return Leaf(classes, y)
    end
    sp = _split(rng, X, y, classes, colnames, cutpoints; max_split_candidates)
    if isnothing(sp) || length(y) ≤ min_data_in_leaf
        return Leaf(classes, y)
    end
    depth += 1

    left = let
        _X, yl = _view_X_y!(mask, X, y, sp, <)
        _tree!(rng, mask, _X, yl, classes, colnames; cutpoints, depth, max_depth)
    end
    right = let
        _X, yr = _view_X_y!(mask, X, y, sp, ≥)
        _tree!(rng, mask, _X, yr, classes, colnames; cutpoints, depth, max_depth)
    end
    node = Node(sp, left, right)
    return node
end

_predict(leaf::Leaf, x::AbstractVector) = leaf.probabilities

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
struct StableForest{T} <: StableModel
    trees::Vector{Union{Node,Leaf}}
    classes::Vector{T}
end
_elements(model::StableForest) = model.trees

"Increase the state of `rng` by `i`."
_change_rng_state!(rng::AbstractRNG, i::Int) = seed!(rng, i)

"""
Return an unique and sorted vector of classes based on `y`.
The vector is sorted to ensure that class ordering is the same between cross-validations.
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
        X::AbstractMatrix,
        y::AbstractVector,
        colnames::Vector{String};
        partial_sampling::Real=PARTIAL_SAMPLING_DEFAULT,
        n_trees::Int=N_TREES_DEFAULT,
        max_depth::Int=MAX_DEPTH_DEFAULT,
        q::Int=10,
        min_data_in_leaf::Int=5
    )
    if 2 < max_depth
        error("Tree depth is too high. Rule filtering for a higher depth is not implemented.")
    end
    if max_depth < 1
        error("Minimum tree depth is 1; got $max_depth")
    end
    # It is essential for the stability to determine the cutpoints over the whole dataset.
    cutpoints = _cutpoints(X, q)
    classes = _classes(y)

    max_split_candidates = round(Int, sqrt(_p(X)))
    n_samples = floor(Int, partial_sampling * length(y))

    trees = Vector{Union{Node,Leaf}}(undef, n_trees)
    Threads.@threads for i in 1:n_trees
        _rng = copy(rng)
        _change_rng_state!(_rng, i)
        # Don't change this to sampling without replacement.
        # When doing that at DecisionTree.jl, the accuracy decreases.
        rows = rand(_rng, 1:length(y), n_samples)
        _X = view(X, rows, :)
        _y = view(y, rows)
        mask = Vector{Bool}(undef, length(y))
        tree = _tree!(
            _rng,
            mask,
            _X,
            _y,
            classes,
            colnames;
            max_split_candidates,
            max_depth,
            q,
            cutpoints,
            min_data_in_leaf
        )
        trees[i] = tree
    end
    return StableForest(trees, classes)
end

function _forest(rng::AbstractRNG, X, y; kwargs...)
    if !(X isa AbstractMatrix || Tables.istable(X))
        error("Input `X` doesn't satisfy the Tables.jl interface.")
    end
    # Tables doesn't assume the data fits in memory so that complicates things a lot.
    # Implementing out-of-memory trees is a problem for later.
    return _forest(rng, matrix(X), y, _colnames(X); kwargs...)
end

function _isempty_error(::StableForest)
    throw(AssertionError("The forest contains no trees"))
end

function _predict(forest::StableForest, row::AbstractVector)
    isempty(_elements(forest)) && _isempty_error(forest)
    probs = [_predict(tree, row) for tree in forest.trees]
    return _median(probs)
end

function _predict(model::StableModel, X::AbstractMatrix)
    probs = [_predict(model, row) for row in eachrow(X)]
    P = reduce(hcat, probs)'
    return UnivariateFinite(model.classes, P; pool=missing)
end

function _predict(model::StableModel, X)
    if !Tables.istable(X)
        error("Expected a Table but got $(typeof(X))")
    end
    return _predict(model, Tables.matrix(X))
end
