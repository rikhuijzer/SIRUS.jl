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

_rough_cutpoint_index_estimate(n::Int, quantile::Real) = Int(floor(quantile * (n + 1)))

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

"""
Return a view on all `y` for which the `comparison` holds in `X[:, feature]`.
"""
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
        data = view(X, :, feature)
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
    return SplitPoint(best_score_feature, best_score_cutpoint)
end

struct Leaf{T}
    majority::T
    values::AbstractVector{T}
end

struct Node{T}
    splitpoint::SplitPoint
    left::Union{Node{T}, Leaf{T}}
    right::Union{Node{T}, Leaf{T}}
end

children(node::Node) = [node.left, node.right]
nodevalue(node::Node) = node.splitpoint

function tree(
        rng::AbstractRNG,
        X,
        y::AbstractVector;
        max_depth=2,
        q=10
    )
    classes = unique(y)
    cutpoints = _cutpoints(X, q)
end
tree(X, y::AbstractVector; kwargs...) = tree(default_rng(), X, y; kwargs...)

end # module
