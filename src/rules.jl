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
        val = round(splitpoint.value; sigdigits=3)
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

_mean_probabilities(V::AbstractVector) = round.(only(mean(V; dims=1)); digits=3)

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

function _rules(forest::Forest)
    rules = Rule[]
    for tree in forest.trees
        tree_rules = _rules!(tree)
        rules = [rules; tree_rules]
    end
    return rules
end

function Base.:(==)(a::SplitPoint, b::SplitPoint)
    return a.feature == b.feature && a.value == b.value
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

function _count_unique_paths(V::Vector{Rule})
    U = unique(V)
    l = length(U)
    counts = Vector{Int}(undef, l)
    for i in 1:l
        u = U[i]
        count = 0
        for v in V
            if v.path == u.path
                count += 1
            end
        end
        counts[i] = count
    end
    return Dict{Rule,Int}(zip(U, counts))
end

"""
Select rules based on frequency of occurence.
`p0` sets a threshold on the number of occurences of a rule.
Below this threshold, the rule is removed.
The default value is based on Figure 4 and 5 of the SIRUS paper.
"""
function _select_rules(rules::Vector{Rule}; p0=20)
    unique_counts = _count_unique_paths(rules)
    for rule in keys(unique_counts)
        if unique_counts[rule] < p0
            delete!(unique_counts, rule)
        end
    end
    return collect(keys(unique_counts))
end

"Filter all rules that have one constraint and are identical to a previous rule with the sign reversed."
function _filter_reversed(rules::Vector{Rule})
    out = copy(rules)
    for rule in rules
        path = rule.path
        splits = path.splits
        if length(splits) == 1
            split = splits[1]
            # Keep the rule with the sign "<" and filter "≥".
            if split.direction == :L
                rev_direction = :R
                rev_split = Split(split.splitpoint, rev_direction)
                rev_path = TreePath([rev_split])
                rev_rule = Rule(rev_path, rule.else_probs, rule.then_probs)
                out = filter!(!=(rev_rule), out)
            end
        end
    end
    return out
end

_n_comparisons(rule::Rule) = length(rule.path.splits)

function _contains(a::Rule, b::Split)
    return any(split -> split.splitpoint == b.splitpoint, a.path.splits)
end

"""
Return `true` if rule `a` and `b` involve the same variables and thresholds.
The sign constraints may be reversed.
"""
function _equal_variables_thresholds(a::Rule, b::Rule)
    if _n_comparisons(a) != _n_comparisons(b)
        return false
    end
    if _n_comparisons(a) == 1 && _n_comparisons(b) == 1
        a_split = a.path.splits[1]
        b_split = b.path.splits[1]
        return a_split.splitpoint == b_split.splitpoint
    end
    @assert _n_comparisons(a) == 2 && _n_comparisons(b) == 2
    matches = map(a.path.splits) do split
        any(b_split -> b_split.splitpoint == split.splitpoint, b.path.splits)
    end
    if all(matches)
        return true
    else
        return false
    end
end


"Return a collection of rules that are linearly dependent."
function _linearly_dependent_rules(rule::Rule, rules::Vector{Rule})
    first_comparison = rule.path.splits[1]
    second_comparison = rule.path.splits[2]
    first_matches = filter(r -> _contains(r, first_comparison), rules)
    isempty(first_matches) && return Rule[]
    second_matches = filter(r -> _contains(r, second_comparison), rules)
    isempty(second_matches) && return Rule[]
    third_matches = filter(r -> _equal_variables_thresholds(r, rule), rules)
    return third_matches
end

"Return the Euclidian distance between the `then_probs` and `else_probs`."
_gap_width(rule::Rule) = norm(rule.then_probs .- rule.else_probs)

"Return true for rules that are a linear duplicate of another more important rule."
function _is_linearly_redundant(rule::Rule, rules::Vector{Rule})
    if _n_comparisons(rule) == 1
        return false
    end
    dependent_rules = _linearly_dependent_rules(rule, rules)
    gap_width = _gap_width(rule)
    return any(r -> gap_width ≤ _gap_width(r) && rule != r, dependent_rules)
end

"Filter all rules that are a linear combination of another rule and have a smaller output gap."
function _filter_linearly_dependent(rules::Vector{Rule})
    return filter(r -> !_is_linearly_redundant(r, rules), rules)
end

"Return post-treated rules."
function _treat_rules(rules::Vector{Rule})
    return _filter_linearly_dependent(_filter_reversed(rules))
end

function _predict(rule::Rule, row::AbstractVector)
    constraints = map(rule.path.splits) do split
        splitpoint = split.splitpoint
        direction = split.direction
        comparison = direction == :L ? (<) : (≥)
        feature = splitpoint.feature
        value = splitpoint.value
        satisfies_constraint = comparison(row[feature], value)
    end
    return all(constraints) ? rule.then_probs : rule.else_probs
end

"""
Predict `y` for a data `row`.
Also returns a vector if the data has only one feature.
"""
function _predict(rules::Vector{Rule}, row::AbstractVector)
    preds = [_predict(rule, row) for rule in rules]
    # For classification, take the average of the rules.
    return only(mean(preds; dims=1))
end
function _predict(rules::Vector{Rule}, x::Union{Tables.MatrixRow, Tables.ColumnsRow})
    return _predict(rules, collect(x))
end
