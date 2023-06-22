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

struct Rule
    path::TreePath
    then::LeafContent
    # Cannot use `else` since it is a reserved keyword
    otherwise::LeafContent
end

_splits(rule::Rule) = rule.path.splits

"""
    feature_names(rule::Rule) -> Vector{String}

Return a vector of feature names; one for each clause in `rule`.
"""
function feature_names(rule::Rule)::Vector{String}
    return String[String(_feature_name(s))::String for s in _splits(rule)]
end

"""
    directions(rule::Rule) -> Vector{Symbol}

Return a vector of split directions; one for each clause in `rule`.
"""
function directions(rule::Rule)::Vector{Symbol}
    return Symbol[_direction(s) for s in _splits(rule)]
end

"""
    values(rule::Rule) -> Vector{Float64}

Return a vector split values; one for each clause in `rule`.
"""
function Base.values(rule::Rule)::Vector{Float64}
    return Float64[Float64(_value(s)) for s in _splits(rule)]
end

"""
    _reverse(rule::Rule) -> Rule

Return a reversed version of the `rule`.
Assumes that the rule has only one split (clause) since two splits
cannot be reversed.
"""
function _reverse(rule::Rule)::Rule
    splits = _splits(rule)
    @assert length(splits) == 1
    split = splits[1]
    path = TreePath([_reverse(split)])
    return Rule(path, rule.otherwise, rule.then)
end

function _left_rule(rule::Rule)::Rule
    splits = _splits(rule)
    @assert length(splits) == 1
    split = splits[1]
    return _direction(split) == :L ? rule : _reverse(rule)
end

function _rules!(
        leaf::Leaf,
        splits::Vector{Split},
        rules::Vector{Rule}
    )::Vector{Rule}
    return push!(rules, rule)
end

function _then_output!(
        leaf::Leaf,
        contents::Vector{LeafContent}
    )::Vector{LeafContent}
    return push!(contents, _content(leaf))
end

"""
Add the leaf contents for the training points which satisfy the
rule to the `contents` vector.
"""
function _then_output!(
        node::Node,
        contents::Vector{LeafContent}
    )::Vector{LeafContent}
    _then_output!(node.left, contents)
    _then_output!(node.right, contents)
    return contents
end

function _else_output!(
        _,
        leaf::Leaf,
        contents::Vector{LeafContent}
    )::Vector{LeafContent}
    return push!(contents, _content(leaf))
end

"""
Add the leaf contents for the training points which do not satisfy
the rule to the `contents` vector.
"""
function _else_output!(
        not_node::Union{Node,Leaf},
        node::Node,
        contents::Vector{LeafContent}
    )::Vector{LeafContent}
    if node == not_node
        return contents
    else
        _else_output!(not_node, node.left, contents)
        _else_output!(not_node, node.right, contents)
        return contents
    end
end

function _frequency_sort(V::AbstractVector)
    counts = _count_unique(V)
    sorted = sort(collect(counts); by=last, rev=true)
    return first.(sorted)
end

function Rule(
        root::Node,
        node::Union{Node, Leaf},
        splits::Vector{Split}
    )::Rule
    path = TreePath(splits)
    then_output = _then_output!(node, Vector{LeafContent}())
    then = _mean(then_output)
    else_output = _else_output!(node, root, Vector{LeafContent}())
    otherwise = _mean(else_output)
    return Rule(path, then, otherwise)
end

function _rules!(
        leaf::Leaf,
        splits::Vector{Split};
        rules::Vector{Rule},
        root::Node
    )::Vector{Rule}
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
    )::Vector{Rule}
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

function _rules(forest::StableForest)::Vector{Rule}
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
    _flip_left(rules::Vector{Rule}) -> Vector{Rule}

Return a subset of `rules` where all rules containing a single clause are flipped to the left.
This is meant to speed up further steps such as finding linearly dependent rules.
"""
function _flip_left(rules::Vector{Rule})::Vector{Rule}
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
Return a sorted subset of `rules` where all the `rule.paths` are unique.
This is done by averaging the `then` and `otherwise` contents.

This is not mentioned in the SIRUS paper, but probably necessary because not sorting the rules by the occurence frequency didn't really affect accuracy.
So, that could mean that the most important rules aren't correct selected which could be caused by multiple paths having different then else probabilities.
"""
function _combine_paths(
        rules::Vector{Rule},
        algo::Algorithm
    )::Vector{Pair{Rule, Int}}
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
        aggregate = algo isa Classification ? _mode : _mean
        then = aggregate(getproperty.(rules, :then))
        otherwise = aggregate(getproperty.(rules, :otherwise))
        combined_rule = Rule(path, then, otherwise)
        averaged_rules[i] = Pair(combined_rule, length(rules)::Int)
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
    return a.path == b.path && a.then == b.then && a.otherwise == b.otherwise
end

function Base.hash(rule::Rule)
    hash([rule.path.splits, rule.then, rule.otherwise])
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
function _process_rules(
        rules::Vector{Rule},
        algo::Algorithm,
        max_rules::Int
    )::Vector{Rule}
    flipped = _flip_left(rules)
    combined = _combine_paths(flipped, algo)
    # This loop is an optimization which manually takes a p0 and checks whether we end up with
    # enough rules. If not, we loop again with more rules.
    for i in 1:3
        required_rule_guess = i^2 * 10 * max_rules
        before = first(combined, required_rule_guess)
        before_rules = first.(before)::Vector{Rule}
        filtered = _filter_linearly_dependent(before_rules)
        too_few = length(filtered) < max_rules
        more_possible = required_rule_guess < length(rules)
        if i < 3 && too_few && more_possible
            continue
        end
        return first(filtered, max_rules)
    end
    @error "This should never happen"
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
    )::StableRules
    processed = _process_rules(rules, algo, model.max_rules)
    weights = _weights(processed, algo, classes, data, outcome, model)
    filtered_rules, filtered_weights = _remove_zero_weights(processed, weights)
    return StableRules(filtered_rules, algo, classes, filtered_weights)
end

function StableRules(
        forest::StableForest,
        data,
        outcome,
        model::Probabilistic,
    )::StableRules
    rules = _rules(forest)
    return StableRules(rules, forest.algo, forest.classes, data, outcome, model)
end

"""
    satisfies(row::AbstractVector, rule::Rule) -> Bool

Return whether data `row` satisfies `rule`.
"""
function satisfies(row::AbstractVector, rule::Rule)::Bool
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

function _predict_rule(rule::Rule, weight::Float16, row::AbstractVector)
    content = satisfies(row, rule) ? rule.then : rule.otherwise
    return weight .* content
end

function _sum(V::AbstractVector{<:AbstractVector})
    M = reduce(hcat, V)
    return [sum(row) for row in eachrow(M)]
end

function _predict(model::StableRules, row::AbstractVector)
    rules_weights = _elements(model)
    isempty(rules_weights) && _isempty_error(model)
    rule_predictions = map(rules_weights) do (rule, weight)
        _predict_rule(rule, weight, row)
    end
    return _sum(rule_predictions)
end
