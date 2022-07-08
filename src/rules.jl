"""
    Split

A split in a tree.
Each rule is based on one or more splits.
"""
struct Split
    splitpoint::SplitPoint
    direction::Symbol # :L or :R
end

function Split(feature::Int, splitval::Float, direction::Symbol)
    return Split(SplitPoint(feature, splitval), direction)
end

_feature(splitpoint::SplitPoint) = splitpoint.feature
_feature(split::Split) = _feature(split.splitpoint)
_value(splitpoint::SplitPoint) = splitpoint.value
_value(split::Split) = _value(split.splitpoint)
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
            feature = parse(Int, c[6:findfirst(']', c)-1])
            splitval = let
                start = direction == :L ? findfirst('<', c) + 1 : findfirst('≥', c) + 3
                parse(Float, c[start:end])
            end
            Split(feature, splitval, direction)
        end
        return TreePath(splits)
    catch e
        msg = """
            Couldn't parse \"$text\"

              Is the syntax correct? Valid examples are:
              - TreePath(" X[i, 1] < 1.0 ")
              - TreePath(" X[i, 1] < 1.0 & X[i, 1] ≥ 4.0 ")

            """
        @error msg exception=(e, catch_backtrace())
    end
end

function Base.show(io::IO, path::TreePath)
    texts = map(path.splits) do split
        splitpoint = split.splitpoint
        comparison = split.direction == :L ? '<' : '≥'
        val = splitpoint.value
        text = "X[i, $(splitpoint.feature)] $comparison $val"
    end
    text = string("TreePath(\" ", join(texts, " & "), " \")")::String
    print(io, text)
end

struct Rule
    path::TreePath
    then_probs::Probabilities
    else_probs::Probabilities
end

_splits(rule::Rule) = rule.path.splits
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

const Probs = Vector{Probabilities}

_then_output!(leaf::Leaf, probs::Probs=Probs()) = push!(probs, leaf.probabilities)

"Return the output average of the training points which satisfy the rule."
function _then_output!(node::Node, probs::Probs=Probs())
    _then_output!(node.left, probs)
    _then_output!(node.right, probs)
    return probs
end

_else_output!(_, leaf::Leaf, probs::Probs) = push!(probs, leaf.probabilities)

"Return the output average of the training points not covered by the rule."
function _else_output!(not_node::Union{Node,Leaf}, node::Node, probs::Probs=Probs())
    if node == not_node
        return probs
    end
    _else_output!(not_node, node.left, probs)
    _else_output!(not_node, node.right, probs)
    return probs
end

function _mean_probabilities(V::AbstractVector{T}) where {T}
    return round.(only(mean(V; dims=1)); digits=3)
end

function Rule(root::Node, node::Union{Node, Leaf}, splits::Vector{Split})
    path = TreePath(splits)
    then_output = _then_output!(node)
    then_probs = _mean_probabilities(then_output)
    else_output = _else_output!(node, root)
    else_probs = _mean_probabilities(else_output)
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
        rules = [rules; tree_rules]
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
        then_probs = _mean_probabilities(getproperty.(rules, :then_probs))
        else_probs = _mean_probabilities(getproperty.(rules, :else_probs))
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
    classes::Vector{T}
    weights::Vector{Float64}
end
_elements(model::StableRules) = zip(model.rules, model.weights)
function _isempty_error(::StableRules)
    throw(AssertionError("The rule model contains no rules"))
end

"""
The weights are regularized slightly since that seems to improve performance.
A higher `penalty` means a stronger regularization, but also lower interpretability of the rules.

!!! note
    Make sure to use enough trees (thousands) for best accuracy.
"""
function _regularize_weights(V::Vector{<:Real}; penalty::Real=0.75)
    @assert 0.0 ≤ penalty ≤ 1.0
    m = mean(V)
    [round(v - (penalty * (v - m)); digits=3) for v in V]
end

function StableRules(rules::Vector{Rule}, classes, max_rules::Int)
    processed = _process_rules(rules, max_rules)
    rules = first.(processed)
    frequencies = last.(processed)
    total = sum(frequencies)
    relative_weights = frequencies ./ total
    weights = _regularize_weights(relative_weights)
    return StableRules(rules, classes, weights)
end

function StableRules(forest::StableForest, max_rules::Int)
    rules = _rules(forest)
    return StableRules(rules, forest.classes, max_rules)
end

function Base.show(io::IO, model::StableRules)
    l = length(model.rules)
    rule_text = string("rule", l == 1 ? "" : "s")
    println(io, "StableRules model with $l $rule_text:")
    for i in 1:l
        ending = i == l ? "" : " +"
        rule = model.rules[i]
        weight = model.weights[i]
        println(io, " $weight * $rule", ending)
    end
    lc = length(model.classes)
    class_text = string("class", lc == 1 ? "" : "es")
    println(io, "and $lc $class_text: $(model.classes)")
end

function _predict(pair::Tuple{Rule,Float64}, row::AbstractVector)
    rule, weight = pair
    constraints = map(rule.path.splits) do split
        splitpoint = split.splitpoint
        direction = split.direction
        comparison = direction == :L ? (<) : (≥)
        feature = splitpoint.feature
        value = splitpoint.value
        satisfies_constraint = comparison(row[feature], value)
    end
    probs = all(constraints) ? rule.then_probs : rule.else_probs
    return weight * probs
end
