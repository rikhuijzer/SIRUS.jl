"""
    SubClause

A subclause denotes a conditional on one feature.
Each rule contains a clause with one or more subclauses.
For example, the rule `if X[i, 1] > 3 & X[i, 2] < 4, then ...` contains two subclauses.

A subclause is equivalent to a split in a decision tree.
In other words, each rule is based on one or more subclauses.
In pratice, a rule is based on at most two subclauses (has at most two subclauses).
The reason for this is that rules with more than two subclauses will not end up
in the final model, as is discussed in the original SIRUS paper.

The data inside a `SubClause` can be accessed via

- `_feature`,
- `_feature_name`,
- `_splitval`, and
- `_direction`.

To obtain the reverse, use `_reverse`.

Note:
this name is not perfect.
A formally better name would be "predicate atom", but that takes more
characters and is also not very intuitive.
Instead, the word `Clause` and `SubClause` seem pretty short and clear.
"""
struct SubClause
    feature::Int
    feature_name::String
    splitval::Float32
    direction::Symbol

    function SubClause(
            feature::Int,
            feature_name::AbstractString,
            splitval::Number,
            direction::Symbol
        )
        @assert direction in (:L, :R)
        return new(feature, String(feature_name), splitval, direction)
    end
end

function SubClause(
        sp::SplitPoint,
        direction::Symbol
    )::SubClause
    return SubClause(sp.feature, sp.feature_name, sp.value, direction)
end

_feature(s::SubClause) = s.feature
_feature_name(s::SubClause) = s.feature_name
_splitval(s::SubClause) = s.splitval
_direction(s::SubClause) = s.direction

function _reverse(s::SubClause)
    direction = s.direction == :L ? :R : :L
    return SubClause(s.feature, s.feature_name, s.splitval, direction)
end

function Base.:(==)(a::SubClause, b::SubClause)
    return a.feature == b.feature && a.feature_name == b.feature_name &&
        a.splitval == b.splitval && a.direction == b.direction
end

"""
    Clause

A clause denotes a conditional on one or more features.
Each rule contains a clause with one or more subclauses.

A clause is equivalent to a path in a decision tree.
For example, the clause `X[i, 1] > 3 & X[i, 2] < 4` can be interpreted as a path
going through two nodes.

In the original SIRUS paper, a path of length `d` is defined as consisting of `d` subclauses.
As discussed above, in practice the number of subclauses or subclauses `d ≤ 2`.

Note that a path can also be a path to a node; not necessarily a leaf.

Data can be accessed via `_subclauses`.
"""
struct Clause
    subclauses::Vector{SubClause}
end

_subclauses(c::Clause) = c.subclauses

function Clause(text::String)
    try
        comparisons = split(strip(text), '&')
        subclauses = map(comparisons) do c
            direction = contains(c, '<') ? :L : :R
            feature_text_end = findfirst(']', c)
            if isnothing(feature_text_end)
                msg = "Couldn't find feature number such as `X[i, 3]` in `$text`"
                throw(ArgumentError(msg))
            end
            feature_text = c[6:feature_text_end - 1]
            if startswith(feature_text, ':')
                msg = "Can only parse feature numbers such as `X[i, 3]`, " *
                    "but got `X[i, $feature_text]`"
                throw(ArgumentError(msg))
            end
            feature = parse(Int, feature_text)
            splitval = let
                start = direction == :L ?
                    findfirst('<', c)::Int + 1 :
                    findfirst('≥', c)::Int + 3
                parse(Float32, c[start:end])
            end
            feature_name = string(feature)::String
            SubClause(feature, feature_name, splitval, direction)
        end
        return Clause(subclauses)
    catch e
        if e isa ArgumentError
            rethrow(e)
        end
        msg = """
            Couldn't parse \"$text\"

              Is the syntax correct? Valid examples are:
              - Clause(" X[i, 1] < 1.0 ")
              - Clause(" X[i, 1] < 1.0 & X[i, 1] ≥ 4.0 ")

            """
        @error msg exception=(e, catch_backtrace())
    end
end

function Base.hash(clause::Clause)
    return hash(clause.subclauses)
end

function Base.:(==)(a::Clause, b::Clause)
    return all(a.subclauses .== b.subclauses)
end

struct Rule
    clause::Clause
    then::LeafContent
    # Cannot use `else` since it is a reserved keyword
    otherwise::LeafContent
end

_clause(rule::Rule) = rule.clause
_subclauses(rule::Rule) = rule.clause.subclauses

"""
    feature_names(rule::Rule) -> Vector{String}

Return a vector of feature names; one for each clause in `rule`.
"""
function feature_names(rule::Rule)::Vector{String}
    return String[String(_feature_name(s))::String for s in _subclauses(rule)]
end

"""
    directions(rule::Rule) -> Vector{Symbol}

Return a vector of split directions; one for each clause in `rule`.
"""
function directions(rule::Rule)::Vector{Symbol}
    return Symbol[_direction(s) for s in _subclauses(rule)]
end

"""
    values(rule::Rule) -> Vector{Float64}

Return a vector split values; one for each subclause in `rule`.
"""
function Base.values(rule::Rule)::Vector{Float64}
    return Float64[Float64(_splitval(s)) for s in _subclauses(rule)]
end

"""
    _reverse(rule::Rule) -> Rule

Return a reversed version of the `rule`.
Assumes that the rule has only one split (clause) since two subclauses
cannot be reversed.
"""
function _reverse(rule::Rule)::Rule
    subclauses = _subclauses(rule)
    @assert length(subclauses) == 1
    subclause = subclauses[1]
    clause = Clause([_reverse(subclause)])
    return Rule(clause, rule.otherwise, rule.then)
end

function _left_rule(rule::Rule)::Rule
    subclauses = _subclauses(rule)
    @assert length(subclauses) == 1
    split = subclauses[1]
    return _direction(split) == :L ? rule : _reverse(rule)
end

function Base.:(==)(a::Rule, b::Rule)
    return a.clause == b.clause && a.then == b.then && a.otherwise == b.otherwise
end

function Base.hash(rule::Rule)
    hash([_subclauses(rule), rule.then, rule.otherwise])
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

function Rule(
        root::Node,
        node::Union{Node, Leaf},
        subclauses::Vector{SubClause}
    )::Rule
    clause = Clause(subclauses)
    then_output = _then_output!(node, Vector{LeafContent}())
    then = _mean(then_output)
    else_output = _else_output!(node, root, Vector{LeafContent}())
    otherwise = _mean(else_output)
    return Rule(clause, then, otherwise)
end

function _rules!(
        leaf::Leaf,
        subclauses::Vector{SubClause};
        rules::Vector{Rule},
        root::Node
    )::Vector{Rule}
    rule = Rule(root, leaf, subclauses)
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
        node::Union{Node, Leaf},
        subclauses::Vector{SubClause}=SubClause[];
        rules::Vector{Rule}=Rule[],
        root::Node=node
    )::Vector{Rule}
    if !isempty(subclauses)
        rule = Rule(root, node, subclauses)
        push!(rules, rule)
    end

    let
        subclause = SubClause(node.splitpoint, :L)
        new_subclauses = [subclause; subclauses]
        _rules!(node.left, new_subclauses; rules, root)
    end

    let
        subclause = SubClause(node.splitpoint, :R)
        new_subclauses = [subclause; subclauses]
        _rules!(node.right, new_subclauses; rules, root)
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

"""
    StableRules{T} <: StableModel

Stable rule-based model containing the rules, algorithm, classes and weights.
This object is created by the MLJ-interface.
"""
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

"""
Simplify the rules that contain a single split by only retaining rules that point left and
removing duplicates.
"""
function _simplify_single_rules(rules::Vector{Rule})::Vector{Rule}
    out = OrderedSet{Rule}()
    for rule in rules
        subclauses = _subclauses(rule)
        if length(subclauses) == 1
            left_rule = _left_rule(rule)
            push!(out, left_rule)
        else
            push!(out, rule)
        end
    end
    return collect(out)
end

"""
Apply _rule selection_ and _rule set post-treatment_
(Bénard et al., [2021](http://proceedings.mlr.press/v130/benard21a)).

We have a slight modification here:
we do not sort first, select some p0 first, and then remove linearly dependent
rules because our linearly dependent filter is quick enough to handle all rules.
This means we don't need to sort at all.

For the sorting, note that the paper talks about sorting by frequency of the
**path** (clause) and not the rule, that is, clause with then and otherwise
probabalities.
"""
function _process_rules(
        rules::Vector{Rule},
        max_rules::Int
    )::Vector{Rule}
    simplified = _simplify_single_rules(rules)::Vector{Rule}
    filtered = _filter_linearly_dependent(simplified)::Vector{Rule}
    return first(filtered, max_rules)
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
        model::Union{Deterministic, Probabilistic}
    )::StableRules
    processed = _process_rules(rules, model.max_rules)
    weights = _weights(processed, algo, classes, data, outcome, model)
    filtered_rules, filtered_weights = _remove_zero_weights(processed, weights)
    return StableRules(filtered_rules, algo, classes, filtered_weights)
end

function StableRules(
        forest::StableForest,
        data,
        outcome,
        model::Union{Deterministic, Probabilistic}
    )::StableRules
    rules = _rules(forest)
    return StableRules(rules, forest.algo, forest.classes, data, outcome, model)
end

"""
    satisfies(row::AbstractVector, rule::Rule) -> Bool

Return whether data `row` satisfies `rule`.
"""
function satisfies(row::AbstractVector, rule::Rule)::Bool
    constraints = map(_subclauses(rule)) do subclause
        comparison = _direction(subclause) == :L ? (<) : (≥)
        feature = _feature(subclause)
        value = _splitval(subclause)
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
