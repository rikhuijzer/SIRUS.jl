struct Regression <: Algorithm end

"""
    RegressionLeaf

Leaf of a decision tree when running regression.
The value is the mean of the `y`'s falling into the region associated with this leaf.
"""
struct RegressionLeaf <: Leaf
    value::Float64
end

function Leaf(::Regression, _, y)
    return RegressionLeaf(mean(y))
end

_predict(leaf::RegressionLeaf, x::AbstractVector) = leaf.value
