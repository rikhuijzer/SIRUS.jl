module ExplainableRules

# Float32 should be precise enough.
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

end # module
