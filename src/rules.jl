"""
    Split

A split in a tree.
Each rule is based on one or more splits.
"""
struct Split
    splitpoint::SplitPoint
    direction::Symbol # :L or :R
end

function Split(feature::Int, name::String, splitval::Float32, direction::Symbol)
    return Split(SplitPoint(feature, splitval, name), direction)
end

_feature(split::Split) = _feature(split.splitpoint)
_value(split::Split) = _value(split.splitpoint)
_feature_name(split::Split) = _feature_name(split.splitpoint)
_direction(split::Split) = split.direction
_reverse(split::Split) = Split(split.splitpoint, split.direction == :L ? :R : :L)

"""
    TreePath

A path of length `d` is defined as consisting of `d` splits.
See SIRUS paper page 434.
Typically, `d ≤ 2`.
Note that a path can also be a path to a node; not necessarily a leaf.
"""
struct TreePath
    splits::Vector{Split}
end

function TreePath(text::String)
    try
        comparisons = split(strip(text), '&')
        splits = map(comparisons) do c
            direction = contains(c, '<') ? :L : :R
            feature_text = c[6:findfirst(']', c) - 1]
            if startswith(feature_text, ':')
                msg = "Can only parse feature numbers such as `X[i, 3]`, " *
                    "but got `X[i, $feature_text]`"
                throw(ArgumentError(msg))
            end
            feature = parse(Int, feature_text)
            splitval = let
                start = direction == :L ? findfirst('<', c) + 1 : findfirst('≥', c) + 3
                parse(Float32, c[start:end])
            end
            feature_name = string(feature)::String
            Split(feature, feature_name, splitval, direction)
        end
        return TreePath(splits)
    catch e
        if e isa ArgumentError
            rethrow(e)
        end
        msg = """
            Couldn't parse \"$text\"

              Is the syntax correct? Valid examples are:
              - TreePath(" X[i, 1] < 1.0 ")
              - TreePath(" X[i, 1] < 1.0 & X[i, 1] ≥ 4.0 ")

            """
        @error msg exception=(e, catch_backtrace())
    end
end

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

struct Rule
    path::TreePath
    then_probs::LeafContent
    else_probs::LeafContent
end

_splits(rule::Rule) = rule.path.splits

"""
    feature_names(rule::Rule) -> Vector{String}

Return a vector of feature names; one for each clause in `rule`.
"""
function feature_names(rule::Rule)::Vector{String}
    return [String(_feature_name(split))::String for split in _splits(rule)]
end

"""
    directions(rule::Rule) -> Vector{Symbol}

Return a vector of split directions; one for each clause in `rule`.
"""
function directions(rule::Rule)::Vector{Symbol}
    return [_direction(split) for split in _splits(rule)]
end

"""
    values(rule::Rule) -> Vector{Float64}

Return a vector split values; one for each clause in `rule`.
"""
function Base.values(rule::Rule)::Vector{Float64}
    return [Float64(_value(split)) for split in _splits(rule)]
end

"""
    _reverse(rule::Rule)

Return a reversed version of the `rule`.
Assumes that the rule has only one split (clause) since two splits
cannot be reversed.
"""
function _reverse(rule::Rule)
    splits = _splits(rule)
    @assert length(splits) == 1
    split = splits[1]
    path = TreePath([_reverse(split)])
    return Rule(path, rule.else_probs, rule.then_probs)
end

function _left_rule(rule::Rule)
    splits = _splits(rule)
    @assert length(splits) == 1
    split = splits[1]
    return _direction(split) == :L ? rule : _reverse(rule)
end

function _rules!(leaf::Leaf, splits::Vector{Split}, rules::Vector{Rule})
    path = TreePath(splits)
    # This assumes that the opposite of a combined rules is the opposite of the last comparison.
    then_probs = leaf.probabilities
    else_probs = 1
    push!(rules, rule)
    return nothing
end

function _then_output!(
        leaf::Leaf,
        probs::Vector{LeafContent}
    )::Vector{LeafContent}
    return push!(probs, _content(leaf))
end

"""
Add the leaf contents for the training points which satisfy the
rule to the `probs` vector.
"""
function _then_output!(
        node::Node,
        probs::Vector{LeafContent}
    )::Vector{LeafContent}
    _then_output!(node.left, probs)
    _then_output!(node.right, probs)
    return probs
end

function _else_output!(
        _,
        leaf::Leaf,
        probs::Vector{LeafContent}
    )::Vector{LeafContent}
    return push!(probs, _content(leaf))
end

"""
Add the leaf contents for the training points which do not satisfy
the rule to the `probs` vector.
"""
function _else_output!(
        not_node::Union{Node,Leaf},
        node::Node,
        probs::Vector{LeafContent}
    )::Vector{LeafContent}
    if node == not_node
        return probs
    end
    _else_output!(not_node, node.left, probs)
    _else_output!(not_node, node.right, probs)
    return probs
end

function _frequency_sort(V::AbstractVector)
    counts = _count_unique(V)
    sorted = sort(collect(counts); by=last, rev=true)
    return first.(sorted)
end

function Rule(root::Node, node::Union{Node, Leaf}, splits::Vector{Split})
    path = TreePath(splits)
    then_output = _then_output!(node, Vector{LeafContent}())
    then_probs = _mean(then_output)
    else_output = _else_output!(node, root, Vector{LeafContent}())
    else_probs = _mean(else_output)
    return Rule(path, then_probs, else_probs)
end

function _rules!(leaf::Leaf, splits::Vector{Split}; rules::Vector{Rule}, root::Node)
    rule = Rule(root, leaf, splits)
    push!(rules, rule)
end

"""
Return a all the rules for all paths in a tree.
This is the rule generation step of SIRUS.
There will be a path for each node and leaf in the tree.
In the paper, for a random free Θ, the list of extracted paths is defined as T(Θ, Dn).
Note that the rules are also created for internal nodes as can be seen from supplement Table 3.
"""
function _rules!(
        node::Node,
        splits::Vector{Split}=Split[];
        rules::Vector{Rule}=Rule[],
        root::Node=node
    )
    if !isempty(splits)
        rule = Rule(root, node, splits)
        push!(rules, rule)
    end

    let
        split = Split(node.splitpoint, :L)
        _splits = [split; splits]
        _rules!(node.left, _splits; rules, root)
    end

    let
        split = Split(node.splitpoint, :R)
        _splits = [split; splits]
        _rules!(node.right, _splits; rules, root)
    end

    return rules
end

function _rules(forest::StableForest)
    rules = Rule[]
    for tree in forest.trees
        tree_rules = _rules!(tree)
        for rule in tree_rules
            push!(rules, rule)
        end
    end
    return rules
end

function Base.hash(path::TreePath)
    return hash(path.splits)
end

"""
Return a subset of `rules` where all rules containing a single clause are flipped to the left.
This is meant to speed up further steps such as finding linearly dependent rules.
"""
function _flip_left(rules::Vector{Rule})
    out = Vector{Rule}(undef, length(rules))
    for i in eachindex(rules)
        rule = rules[i]
        splits = _splits(rule)
        if length(splits) == 1
            left_rule = _left_rule(rule)
            out[i] = left_rule
        else
            out[i] = rule
        end
    end
    return out
end

"""
Return a subset of `rules` where all the `rule.paths` are unique.
This is done by averaging the `then_probs` and `else_probs`.

This is not mentioned in the SIRUS paper, but probably necessary because not sorting the rules by the occurence frequency didn't really affect accuracy.
So, that could mean that the most important rules aren't correct selected which could be caused by multiple paths having different then else probabilities.
"""
function _combine_paths(rules::Vector{Rule})
    U = unique(getproperty.(rules, :path))
    init = zip(U, repeat([Vector{Rule}[]], length(U)))
    duplicate_paths = Dict{TreePath,Vector{Rule}}(init)
    for rule in rules
        push!(duplicate_paths[rule.path], rule)
    end
    averaged_rules = Vector{Pair{Rule,Int}}(undef, length(duplicate_paths))
    for (i, path) in enumerate(keys(duplicate_paths))
        rules = duplicate_paths[path]
        # Taking the mode because that might make more sense here.
        # Doesn't seem to affect accuracy so much.
        then_probs = _median(getproperty.(rules, :then_probs))
        else_probs = _median(getproperty.(rules, :else_probs))
        combined_rule = Rule(path, then_probs, else_probs)
        averaged_rules[i] = Pair(combined_rule, length(rules))
    end
    sorted = sort(averaged_rules; by=last, rev=true)
    return sorted
end

function Base.:(==)(a::SplitPoint, b::SplitPoint)
    return a.feature == b.feature && a.value ≈ b.value
end

function Base.:(==)(a::Split, b::Split)
    return a.direction == b.direction && a.splitpoint == b.splitpoint
end

function Base.:(==)(a::TreePath, b::TreePath)
    return all(a.splits .== b.splits)
end

function Base.:(==)(a::Rule, b::Rule)
    return a.path == b.path && a.then_probs == b.then_probs && a.else_probs == b.else_probs
end

function Base.hash(rule::Rule)
    hash([rule.path.splits, rule.then_probs, rule.else_probs])
end

function _count_unique(V::AbstractVector{T}) where T
    U = unique(V)
    l = length(U)
    counts = Dict{T,Int}(zip(U, zeros(l)))
    for v in V
        counts[v] += 1
    end
    return counts
end

"""
Return a subset of `rules` of length `max_rules`.

!!! note
    This doesn't use p0 like is done in the paper.
    The problem, IMO, with p0 is that it is very difficult to decide beforehand what p0 is suitable and so it requires hyperparameter tuning.
    Instead, luckily, the linearly dependent filter is quite fast here, so passing a load of rules into that and then selecting the first `max_rules` is feasible.
"""
function _process_rules(rules::Vector{Rule}, max_rules::Int)
    flipped = _flip_left(rules)
    combined = _combine_paths(flipped)
    for i in 1:3
        required_rule_guess = i^2 * 10 * max_rules
        before = first(combined, required_rule_guess)
        filtered = _filter_linearly_dependent(before)
        too_few = length(filtered) < max_rules
        more_possible = required_rule_guess < length(rules)
        if i < 3 && too_few && more_possible
            continue
        end
        return first(filtered, max_rules)
    end
end

struct StableRules{T} <: StableModel
    rules::Vector{Rule}
    algo::Algorithm
    classes::Vector{T}
    weights::Vector{Float16}
end
_elements(model::StableRules) = zip(model.rules, model.weights)
function _isempty_error(::StableRules)
    throw(AssertionError("The rule model contains no rules"))
end

function _remove_zero_weights(rules::Vector{Rule}, weights::Vector{Float16})
    filtered_rules = Rule[]
    filtered_weights = Float16[]
    @assert length(rules) == length(weights)
    for i in eachindex(rules)
        if weights[i] != Float16(0.0)
            push!(filtered_rules, rules[i])
            push!(filtered_weights, weights[i])
        end
    end
    return filtered_rules, filtered_weights
end

function StableRules(
        rules::Vector{Rule},
        algo::Algorithm,
        classes,
        data,
        outcome,
        model::Probabilistic
    )
    processed = _process_rules(rules, model.max_rules)
    rules = first.(processed)
    weights = _weights(rules, classes, data, outcome, model)
    filtered_rules, filtered_weights = _remove_zero_weights(rules, weights)
    return StableRules(filtered_rules, algo, classes, filtered_weights)
end

function StableRules(
        forest::StableForest,
        data,
        outcome,
        model::Probabilistic,
    )
    rules = _rules(forest)
    return StableRules(rules, forest.algo, forest.classes, data, outcome, model)
end

"Return only the last result for the binary case because the other is 1 - p anyway."
function _simplify_binary_probabilities(weight, probs::AbstractVector)
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

"Return a pretty formatted so that it is easy to understand."
function _pretty_rule(weight, rule::Rule)
    then_probs = _simplify_binary_probabilities(weight, rule.then_probs)
    else_probs = _simplify_binary_probabilities(weight, rule.else_probs)
    condition = _pretty_path(rule.path)
    return "if $condition then $then_probs else $else_probs"
end

function Base.show(io::IO, model::StableRules)
    l = length(model.rules)
    rule_text = string("rule", l == 1 ? "" : "s")::String
    println(io, "StableRules model with $l $rule_text:")
    for i in 1:l
        ending = i < l ? " +" : ""
        rule = _pretty_rule(model.weights[i], model.rules[i])
        println(io, " $rule", ending)
    end
    C = model.classes
    lc = length(C)
    note = lc == 2 ?
    "\nNote: showing only the probability for class $(last(C)) since class $(first(C)) has probability 1 - p." :
        ""
    println(io, "and $lc classes: $C. $note")
end

"""
    satisfies(row::AbstractVector, rule::Rule)

Return whether data `row` satisfies `rule`.
"""
function satisfies(row::AbstractVector, rule::Rule)
    constraints = map(rule.path.splits) do split
        splitpoint = split.splitpoint
        direction = split.direction
        comparison = direction == :L ? (<) : (≥)
        feature = splitpoint.feature
        value = splitpoint.value
        satisfies_constraint = comparison(row[feature], value)
    end
    return all(constraints)
end

"Return the then or else probabilities for data `row` according to `rule`."
function _probability(row::AbstractVector, rule::Rule)
    return satisfies(row, rule) ? rule.then_probs : rule.else_probs
end

function _predict(pair::Tuple{Rule, Float16}, row::AbstractVector)
    rule, weight = pair
    probs = _probability(row, rule)
    return weight .* probs
end

function _sum(V::AbstractVector{<:AbstractVector})
    M = reduce(hcat, V)
    return [sum(row) for row in eachrow(M)]
end

function _predict(model::StableRules, row::AbstractVector)
    isempty(_elements(model)) && _isempty_error(model)
    probs = _predict.(_elements(model), Ref(row))
    return _sum(probs)
end
