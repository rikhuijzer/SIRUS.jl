"""
Convert the dataset such that each rule becomes a binary feature.

Using binary features and not the probabilities because the probabilities have multiple dimensions, I think.
"""
function _binary_features(rules::Vector{Rule}, data)
    if Tables.istable(data)
        data = Tables.matrix(data)
    end
    h = first(size(data))
    X = Matrix{Float16}(undef, h, length(rules))
    for (col, rule) in enumerate(rules)
        for row_index in 1:h
            row = data[row_index, :]
            X[row_index, col] = satisfies(row, rule) ? 1.0 : 0.0
        end
    end
    return X
end

"Return using min-max normalization to scale the data to [0, 1]."
function _normalize!(X::Vector{T}) where {T<:Real}
    lower = minimum(X)
    upper = maximum(X)
    @inbounds @simd for i in eachindex(X)
        X[i] = (X[i] - lower) / (upper - lower)
    end
    return X
end

"""
Return the coefficients obtained when fitting a regularized linear model over the binary features.
The `lambda` specifies the strength of the L2 (ridge) regression.

For example, the data for fitting could have the following variables:

| Rule 1 | Rule 2 | Rule 3 | Outcome |
|--------|--------|--------|---------|
| ...... | ...... | ...... | ....... |

The regression then finds a coeffient for each rule which is based on how much
each rule is associated with the outcome. I don't know why this makes sense,
but based on benchmarks it does.

Note that the exact weights do not matter for the classification case, since
the highest class will be selected anyway. For regression however, the weights
should sum to roughly one.
"""
function _estimate_coefficients(
        algo::Algorithm,
        binary_feature_data::Matrix{Float16},
        outcome::Vector{Float16},
        model::Probabilistic
    )
    # Lasso and ridge are non-invariant and thus require normalized data.
    _normalize!(outcome)

    # Code is based on the definition for MMI.fit inside MLJLinearModels.jl.
    # Using Ridge because it allows an analytical solver.
    # Also, ElasticNet shows no clear benefit in accuracy.
    # According to Clement, avoid Lasso since it would introduce additional
    # sparsity and then instability in the rule selection.
    model = RidgeRegressor(; fit_intercept=false, model.lambda)
    coefs = MLJLinearModels.fit(glr(model), binary_feature_data, outcome)::Vector
    if algo isa Regression
        # Ensure that coefs sum roughly to one.
        total = sum(coefs)
        coefs = coefs ./ total
    end
    # Avoid negative coefficients.
    return max.(coefs, 0)
end

"""
Return the weights which are regularized to improve performance.

!!! note
    This step requires many trees (thousands) for best accuracy.
"""
function _weights(
        rules::Vector{Rule},
        algo::Algorithm,
        classes::AbstractVector,
        data,
        outcome::AbstractVector,
        model::Probabilistic
    )
    binary_feature_data = _binary_features(rules, data)
    y = convert(Vector{Float16}, outcome)
    coefficients = _estimate_coefficients(algo, binary_feature_data, y, model)
    return coefficients::Vector{Float16}
end
