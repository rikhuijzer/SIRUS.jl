module StableTrees

import AbstractTrees: children, nodevalue

using Random: AbstractRNG, default_rng

const Float = Float32

"""
    gini(y::AbstractVector, classes::AbstractVector)

Return the Gini index for a vector outcomes `y` and `classes`.
Here, `y` is usually a view on the outcome values in some region.
Inside that region, `gini` is a measure of node purity.
If all values in the region have the same class, then gini is zero.
"""
function gini(y::AbstractVector, classes::AbstractVector)
    proportions = Vector{Float}(undef, length(classes))
    len_y = length(y)
    for (i, class) in enumerate(classes)
        proportion = count(y .== class) / len_y
        proportions[i] = proportion
    end
    return sum(proportions .* (1 .- proportions))
end

"""
Return a rough estimate for the index of the cutpoint.
Choose the highest suitable index if there is more than one suitable index.
The reason is that this will split the data nicely in combination with the `<` used later on.
For example, for [1, 2, 3, 4], both 2 and 3 satisfy the 0.5 quantile.
In this case, we pick the ceil, so 3.
Next, the tree will split on 3, causing left (<) to contain 1 and 2 and right (≥) to contain 3 and 4.
"""
_rough_cutpoint_index_estimate(n::Int, quantile::Real) = Int(ceil(quantile * (n + 1)))

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
    quantiles = range(0.0; stop=1.0, length=q)
    return Float[_empirical_quantile(V, quantile) for quantile in quantiles]
end

"Return the number of features `p` in a dataset `X`."
_p(X) = size(X, 2)

"Set of possible cutpoints, that is, numbers from the empirical quantiles."
const Cutpoints = Matrix{Float}

"Return a matrix containing `q` rows and one column for each feature in the dataset."
function _cutpoints(X, q::Int)
    p = _p(X)
    cutpoints = Cutpoints(undef, q, p)
    for feature in 1:p
        V = view(X, :, feature)
        cutpoints[:, feature] = _cutpoints(V, q)
    end
    return cutpoints
end

"Return a view on all `y` for which the `comparison` holds in `X[:, feature]`."
function _view_y(X, y, feature::Int, comparison, cutpoint)
    indexes_in_region = comparison.(X[:, feature], cutpoint)
    return view(y, indexes_in_region)
end

"Location where the tree splits for some split."
struct SplitPoint
    feature::Int
    value::Float
end

"""
Return the split for which the gini index is minimized.
This function receives the cutpoints for the whole dataset `D` because `X` can be a subset of `D`.
"""
function _split(
        X,
        y::AbstractVector,
        classes::AbstractVector,
        cutpoints::Cutpoints
    )
    best_score = Float(999)
    best_score_feature = 0
    best_score_cutpoint = Float(0)
    for feature in 1:_p(X)
        feature_cutpoints = view(cutpoints, :, feature)
        for cutpoint in feature_cutpoints
            gini_left = gini(_view_y(X, y, feature, <, cutpoint), classes)
            gini_right = gini(_view_y(X, y, feature, ≥, cutpoint), classes)
            score = gini_left + gini_right
            if score < best_score
                best_score = score
                best_score_feature = feature
                best_score_cutpoint = cutpoint
            end
        end
    end
    if best_score == Float(999)
        return nothing
    end
    return SplitPoint(best_score_feature, best_score_cutpoint)
end

struct Leaf{T}
    majority::T
    values::AbstractVector{T}
end

function _mode(y::AbstractVector{<:Real})
    @assert !isempty(y)
    U = unique(y)
    counts = Dict(zip(U, zeros(Int, length(U))))
    for value in y
        counts[value] += 1
    end
    max_counted_value = y[1]
    max_count = 0
    for key in keys(counts)
        count = counts[key]
        if max_count < count
            max_counted_value = key
            max_count = count
        end
    end
    return max_counted_value
end

function Leaf(y::AbstractVector{<:Real})
    majority = _mode(y)
    return Leaf{eltype(y)}(majority, y)
end

struct Node{T}
    splitpoint::SplitPoint
    left::Union{Node{T}, Leaf{T}}
    right::Union{Node{T}, Leaf{T}}
end

children(node::Node) = [node.left, node.right]
nodevalue(node::Node) = node.splitpoint

"Return a view on all rows in `X` and `y` for which the `comparison` holds in `X[:, feature]`."
function _view_X_y(X, y, splitpoint::SplitPoint, comparison)
    indexes_in_region = comparison.(X[:, splitpoint.feature], splitpoint.value)
    X_view = view(X, indexes_in_region, :)
    y_view = view(y, indexes_in_region)
    return (X_view, y_view)
end

function _verify_lengths(X, y)
    l1 = size(X, 1)
    l2 = length(y)
    if l1 != l2
        error("Expected X and y to have the same number of rows, but got $l1 and $l2 rows")
    end
end

function _tree(
        rng::AbstractRNG,
        X,
        y::AbstractVector;
        depth=0,
        max_depth=2,
        q=10,
        cutpoints::Cutpoints=_cutpoints(X, q),
        classes=unique(y),
        min_data_in_leaf=5
    )
    _verify_lengths(X, y)
    if depth == max_depth
        return Leaf(y)
    end
    splitpoint = _split(X, y, classes, cutpoints)
    if isnothing(splitpoint)
        return Leaf(y)
    end
    depth += 1
    left = let
        _X, _y = _view_X_y(X, y, splitpoint, <)
        _tree(rng, _X, _y; cutpoints, classes, depth)
    end
    right = let
        _X, _y = _view_X_y(X, y, splitpoint, ≥)
        _tree(rng, _X, _y; cutpoints, classes, depth)
    end
    node = Node{eltype(y)}(splitpoint, left, right)
    return node
end
_tree(X, y::AbstractVector; kwargs...) = _tree(default_rng(), X, y; kwargs...)

end # module
