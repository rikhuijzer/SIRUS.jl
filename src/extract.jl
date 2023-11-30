
"Estimate the importance of a rule."
function _rule_importance(weight::Number, rule::Rule)
    # TODO: THIS SHOULD USE THE GAP SIZE FUNCTION.
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
The aim is to satisfy the following property, so that the features can be
ordered by importance:

> Given two features A and B, if A has more effect on the outcome, then
> feature_importance(model, A) > feature_importance(model, B).

!!! note
    This function provides only an importance _estimate_ because the effect on
    the outcome depends on the data, and because it doesn't take into account
    that a feature can have a lower effect if it is in a clause together with
    another subclause.
"""
function feature_importance(
        model::StableRules,
        feature_name::String
    )
    importance = 0.0
    found_feature = false
    for (i, rule) in enumerate(model.rules)
        for subclause::SubClause in _subclauses(rule)
            if _feature_name(subclause)::String == feature_name
                found_feature = true
                weight = model.weights[i]
                importance += _rule_importance(weight, rule)
            end
        end
    end
    if !found_feature
        throw(KeyError("Feature `$feature_name` not found in the model."))
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

"""
    feature_importances(
        models::Union{StableRules, Vector{StableRules}}
        feature_names
    )::Vector{NamedTuple{(:feature_name, :importance), Tuple{String, Float64}}}

Return the feature names and importances, sorted by feature importance in descending order.
"""
function feature_importances(
        models::Union{StableRules, Vector{StableRules}},
        feature_names::Vector{String}
    )::Vector{NamedTuple{(:feature_name, :importance), Tuple{String, Float64}}}
    @assert length(unique(feature_names)) == length(feature_names)
    importances = map(feature_names) do feature_name
        importance = feature_importance(models, feature_name)
        (; feature_name, importance)
    end
    alg = Helpers.STABLE_SORT_ALG
    return sort(importances; alg, by=last, rev=true)
end

function feature_importances(
        models::Union{StableRules, Vector{StableRules}},
        feature_names
    )::Vector{NamedTuple{(:feature_name, :importance), Tuple{String, Float64}}}
    return feature_importances(models, string.(feature_names))
end
