module ExplainableRules

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

struct Score{T}
    feature::Int
    cutpoint::T
    value::Float
end

"""
Return a view on all `y` for which the `comparison` holds in `X[:, feature]`.
"""
function _view_y(X, y, feature::Int, comparison, cutpoint)
    indexes_in_region = comparison.(X[:, feature], cutpoint)
    return view(y, indexes_in_region)
end

"Return the best split by picking the lowest score."
function _best_split(scores::Vector{Score{T}})::Tuple{Int,T} where T
    best_score_index = 0
    best_score_value = 999.0f0
    for i in 1:length(scores)
        score = scores[i]
        if score.value < best_score_value
            best_score_index = i
        end
    end
    best_score = scores[best_score_index]
    return (best_score.feature, best_score.cutpoint)
end

"Return the split for which the gini index is minimized."
function _find_split(X, y::AbstractVector, classes::AbstractVector)
    best_score = Float(999)
    best_score_feature = 0
    best_score_cutpoint = eltype(X)(0)
    for feature in 1:size(X, 2)
        U = unique(X[:, feature])
        for value in U
            cutpoint = value
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
    return best_score_feature, best_score_cutpoint
end

function build_tree(rng::AbstractRNG, X, y::AbstractVector)
    classes = unique(y)
    
end
build_tree(X, y::AbstractVector) = build_tree(default_rng(), X, y)

end # module
