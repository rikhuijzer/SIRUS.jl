struct Classification <: Algorithm end

"""
    _gini(y::AbstractVector, classes::AbstractVector)

Return the Gini index for a vector outcomes `y` and `classes`.
Here, `y` is usually a view on the outcome values in some region.
Inside that region, `gini` is a measure of node purity.
If all values in the region have the same class, then gini is zero.
The equation is mentioned on Wikipedia as
``1 - \\sum{class \\in classes} p_i^2,``
where ``p_i`` denotes the fraction (proportion) of items labeled
with class ``i`` in the set.
"""
function _gini(y::AbstractVector, classes)
    len_y = length(y)
    impurity = 1.0

    for class in classes
        c = _count_equal(y, class)
        proportion = c / len_y
        impurity -= proportion^2
    end
    return impurity
end

function _information_gain(
        y::AbstractVector,
        yl::AbstractVector,
        yr::AbstractVector,
        classes,
        starting_impurity::Real
    )::Real
    p = length(yl) / length(y)
    impurity_change = p * _gini(yl, classes) + (1 - p) * _gini(yr, classes)
    return starting_impurity - impurity_change
end

"Data to be re-used in the loop on features and splitpoints."
function _reused_data(
        ::Classification,
        y::AbstractVector,
        classes
    )::Real
    return _gini(y, classes)
end

"Return the start score for the maximization problem."
_start_score(::Classification) = 0.0

function _current_score(
        ::Classification,
        y::AbstractVector,
        vl::AbstractVector,
        vr::AbstractVector,
        classes,
        reused_data::Real
    )::Real
    return _information_gain(y, vl, vr, classes, reused_data)
end

function _score_improved(
    ::Classification,
    best_score::Real,
    current_score::Real
)::Bool
    return best_score ≤ current_score
end

"""
    ClassificationLeaf

Leaf of a decision tree when running classification.
The probabilities are based on the `y`'s falling into the region associated with this leaf.
The meaning of each index in the probabilities vector is given by the `classes` vector.
"""
struct ClassificationLeaf <: Leaf
    values::ClassificationValues
end

function ClassificationLeaf(values::AbstractVector)
    return ClassificationLeaf(ClassificationValues(values))
end

function Leaf(::Classification, classes, y)
    l = length(y)
    probabilities = [_count_equal(y, c) / l for c in classes]
    # Not creating a UnivariateFinite because it requires MLJBase
    return ClassificationLeaf(probabilities)
end

_predict(leaf::ClassificationLeaf, x::AbstractVector) = leaf.values.probabilities
