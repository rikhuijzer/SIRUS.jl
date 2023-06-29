struct Regression <: Algorithm end

"Data to be re-used in the loop on features and splitpoints."
function _reused_data(
        ::Regression,
        _::AbstractVector,
        _
    )::Real
    return 0
end

function _rss(y::AbstractVector)::Real
    m = mean(y)
    out = 0.0
    @inbounds @simd for x in y
        out += (x - m)^2
    end
    return out
end

"Return the start score for the minimization problem."
_start_score(::Regression) = Inf

function _current_score(
        ::Regression,
        y::AbstractVector,
        vl::AbstractVector,
        vr::AbstractVector,
        _,
        _::Real
    )::Real
    return _rss(vl) + _rss(vr)
end

function _score_improved(
    ::Regression,
    best_score::Real,
    current_score::Real
)::Bool
    return current_score < best_score
end

"""
    RegressionLeaf

Leaf of a decision tree when running regression.
The value is the mean of the `y`'s falling into the region associated with this leaf.
"""
struct RegressionLeaf <: Leaf
    value::Float64
end

_content(leaf::RegressionLeaf)::LeafContent = Float64[leaf.value]

function Leaf(::Regression, _, y)
    return RegressionLeaf(mean(y))
end

_predict(leaf::RegressionLeaf, x::AbstractVector) = leaf.value
