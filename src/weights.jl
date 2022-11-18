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
    _estimate_coefficients(binary_feature_data, outcome, model)

Return the coefficients obtained when fitting a regularized linear model over the binary features.
The `lambda` specifies the strength of the L2 (ridge) regression and `gamma` the strenght of the L1 (lasso) regression.
"""
function _estimate_coefficients(binary_feature_data, outcome, model)
    # Code is based on the definition for MMI.fit inside MLJLinearModels.jl.
    model = ElasticNetRegressor(; fit_intercept=false, model.lambda, model.gamma)
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
        outcome::AbstractVector,
        model
    )
    binary_feature_data = _binary_features(rules, data)
    coefficients = _estimate_coefficients(binary_feature_data, outcome, model)
    return coefficients::Vector{Float64}
end
