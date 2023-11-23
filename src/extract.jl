"Estimate the importance of a rule."
function _rule_importance(weight::Number, rule::Rule)
    importance = 0.0
    thens = rule.then::Vector{Float64}
    otherwises = rule.otherwise::Vector{Float64}
    @assert length(thens) == length(otherwises)
    n_classes = length(thens)
    for (then, otherwise) in zip(thens, otherwises)
        importance += weight * abs(then - otherwise)
    end
    return importance / n_classes
end

"""
    feature_importance(
        models::Union{StableRules, Vector{StableRules}},
        feature_name::AbstractString
    )

Estimate the importance of the given `feature_name`.
The aim is to satisfy the following property:

> Given two features X and Y, if X has more effect on the outcome, then
> feature_importance(model, X) > feature_importance(model, Y).

!!! note
    This function provides only an importance _estimate_ because
    the effect on the outcome depends on the data.
"""
function feature_importance(
        model::StableRules,
        feature_name::String
    )
    importance = 0.0
    for (i, rule) in enumerate(model.rules)
        for subclause::SubClause in _subclauses(rule)
            if _feature_name(subclause)::String == feature_name
                weight = model.weights[i]
                importance += _rule_importance(weight, rule)
            end
        end
    end
    return importance
end

function feature_importance(model::StableRules, feature_name::AbstractString)
    return feature_importance(model, string(feature_name)::String)
end

function feature_importance(
        models::Vector{<:StableRules},
        feature_name::String
    )
    importance = 0.0
    for model in models
        importance += feature_importance(model, feature_name)
    end
    return importance / length(models)
end

function feature_importance(models::Vector{<:StableRules}, feature_name::AbstractString)
    return feature_importance(models, string(feature_name)::String)
end
