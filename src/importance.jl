
"Estimate the importance of a rule."
function _rule_importance(weight::Number, rule::Rule)
    importance = 0.0
    gap = gap_size(rule)
    n_classes = length(rule.then)
    return (weight * gap) / n_classes
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
> `feature_importance(model, A) > feature_importance(model, B)`.

This is based on the [`gap_size`](@ref) function. The gap size is the
difference between the then and otherwise (else) probabilities. A smaller gap
size implies a smaller CART-splitting criterion, which implies a smaller
occurrence frequency (see the appendix at
<https://proceedings.mlr.press/v130/benard21a.html> for an example).

!!! note
    This function provides only an importance _estimate_ because the effect on
    the outcome depends on the data.
"""
function feature_importance(
        model::StableRules,
        feat_name::String
    )
    importance = 0.0
    found_feature = false
    for (i, rule) in enumerate(model.rules)
        for subclause::SubClause in subclauses(rule)
            if feature_name(subclause)::String == feat_name
                found_feature = true
                weight = model.weights[i]
                importance += _rule_importance(weight, rule)
            end
        end
    end
    if !found_feature
        throw(ArgumentError("Feature `$feat_name` not found in the model."))
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
        feat_names::Vector{String}
    )::Vector{NamedTuple{(:feature_name, :importance), Tuple{String, Float64}}}

Return the feature names and importances, sorted by feature importance in descending order.
"""
function feature_importances(
        models::Union{StableRules, Vector{<:StableRules}},
        feat_names::Vector{String}
    )::Vector{NamedTuple{(:feature_name, :importance), Tuple{String, Float64}}}
    @assert length(unique(feat_names)) == length(feat_names)
    importances = map(feat_names) do feat_name
        importance = feature_importance(models, feat_name)
        (; feature_name=feat_name, importance)
    end
    alg = Helpers.STABLE_SORT_ALG
    return sort(importances; alg, by=last, rev=true)
end

function feature_importances(
        models::Union{StableRules, Vector{<:StableRules}},
        feature_names
    )::Vector{NamedTuple{(:feature_name, :importance), Tuple{String, Float64}}}
    return feature_importances(models, string.(feature_names))
end
