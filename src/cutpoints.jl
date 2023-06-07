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
function _cutpoints(X, q::Int)::Vector{Cutpoints}
    p = _p(X)
    cutpoints = Vector{Cutpoints}(undef, p)
    for feature in 1:p
        V = _view_feature(X, feature)
        cutpoints[feature] = unique(_cutpoints(V, q))
    end
    return cutpoints
end
