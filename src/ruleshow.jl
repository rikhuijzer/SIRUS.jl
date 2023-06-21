"""
Return a feature name that can be shown as `[:, 1]` or `[:, :some_var]`.
"""
function _pretty_feature_name(feature::Int, feature_name::String255)
    name = String(feature_name)::String
    s = string(feature)::String
    if s == name
        return s
    else
        return string(':', name)
    end
end

function _pretty_path(path::TreePath)
    texts = map(path.splits) do split
        sp = split.splitpoint
        comparison = split.direction == :L ? '<' : '≥'
        val = sp.value
        feature = _pretty_feature_name(sp.feature, sp.feature_name)
        text = "X[i, $feature] $comparison $val"
    end
    return join(texts, " & ")
end

function Base.show(io::IO, path::TreePath)
    text = string("TreePath(\" ", _pretty_path(path), " \")")::String
    print(io, text)
end

"Return only the last result for the binary case because the other is 1 - p anyway."
function _simplify_binary_probabilities(
        weight,
        probs::LeafContent
    )
    if length(probs) == 2
        left = first(probs)
        right = last(probs)
        if !isapprox(left + right, 1.0; atol=0.01)
            @warn """
                The sum of the two probabilities $probs doesn't add to 1.
                This is unexpected.
                Please open an issue at SIRUS.jl.
                """
        end
        return round(weight * right; digits=3)
    else
        return round.(weight .* probs; digits=3)
    end
end

function _simplify_regression_contents(
        weight,
        contents::LeafContent
    )
    content = only(contents)
    return round(weight * content; digits=3)
end

"Return a pretty formatted so that it is easy to understand."
function _pretty_rule(algo::Algorithm, weight, rule::Rule)
    simplify = algo isa Classification ?
        _simplify_binary_probabilities :
        _simplify_regression_contents
    then = simplify(weight, rule.then)
    otherwise = simplify(weight, rule.otherwise)
    condition = _pretty_path(rule.path)
    return "if $condition then $then else $otherwise"
end

function Base.show(io::IO, model::StableRules)
    l = length(model.rules)
    rule_text = string("rule", l == 1 ? "" : "s")::String
    println(io, "StableRules model with $l $rule_text:")
    for i in 1:l
        ending = i < l ? " +" : ""
        rule = _pretty_rule(model.algo, model.weights[i], model.rules[i])
        println(io, " $rule", ending)
    end
    if model.algo isa Classification
        C = model.classes
        lc = length(C)
        note = lc == 2 ?
        "\nNote: showing only the probability for class $(last(C)) since class $(first(C)) has probability 1 - p." :
            ""
        println(io, "and $lc classes: $C. $note")
    end
end
