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
Return the coefficients obtained when fitting a regularized linear model over the binary features.
The `lambda` specifies the strength of the L2 (ridge) regression.
"""
function _estimate_coefficients(
        binary_feature_data::Matrix{Float16},
        outcome::Vector{Float16},
        model::Probabilistic
    )
    # Code is based on the definition for MMI.fit inside MLJLinearModels.jl.
    # Using Ridge because it allows an analytical solver.
    # Also, ElasticNet shows no clear benefit in accuracy.
    model = RidgeRegressor(; fit_intercept=false, model.lambda)
    return MLJLinearModels.fit(glr(model), binary_feature_data, outcome)::Vector
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
    y = convert(Vector{Float16}, outcome)
    coefficients = _estimate_coefficients(binary_feature_data, y, model)
    return coefficients::Vector{Float16}
end
