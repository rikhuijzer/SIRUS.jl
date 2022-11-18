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

"""
    _estimate_coefficients(binary_feature_data, outcome)

Return the coefficients obtained when fitting a regularized linear model over the binary features.
"""
function _estimate_coefficients(binary_feature_data, outcome)
    # L2
    lambda = 0.4
    # L1 (produces sparse models, that is, coefficients are pulled to zero.)
    gamma = 0.2
    # Code is based on the definition for MMI.fit inside MLJLinearModels.jl.
    model = ElasticNetRegressor(; fit_intercept=false)
    y = convert(Vector{Float16}, outcome)
    return MLJLinearModels.fit(glr(model), binary_feature_data, y)::Vector
end

"""
Return the weights which are regularized to improve performance.

!!! note
    Make sure to use enough trees (thousands) for best accuracy.
"""
function _weights(
        rules::Vector{Rule},
        classes::AbstractVector,
        data,
        outcome::AbstractVector
    )
    binary_feature_data = _binary_features(rules, data)
    coefficients = _estimate_coefficients(binary_feature_data, outcome)
    @show coefficients
    return coefficients::Vector{Float64}
end
