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
            Couldn't parse $text. Is the syntax correct? Valid examples are:
            - TreePath(" X[i, 1] < 1.0 ")
            - TreePath(" X[i, 1] < 1.0 & X[i, 1] ≥ 4.0 ")
            """
        @error msg exception=(e, catch_backtrace())
    end
end

function show(io::IO, path::TreePath)
    texts = map(path.splits) do split
        splitpoint = split.splitpoint
        comparison = split.direction == :L ? '<' : '≥'
        text = "X[i, $(splitpoint.feature)] $comparison $(splitpoint.value)"
    end
    text = string("TreePath(\" ", join(texts, " & "), " \")")::String
    print(io, text)
end

function _paths!(leaf::Leaf, splits::Vector{Split}, paths::Vector{TreePath})
    path = TreePath(splits)
    push!(paths, path)
    return nothing
end

"""
Return a set of all the paths in the `node` which can be a tree.
This is the rule generation step of SIRUS.
There will be a path for each node and leaf in the tree.
In the paper, for a random free Θ, the list of extracted paths is defined as T(Θ, Dn).
"""
function _paths!(
        node::Node,
        splits::Vector{Split}=Split[],
        paths::Vector{TreePath}=TreePath[]
    )
    if !isempty(splits)
        path = TreePath(splits)
        push!(paths, path)
    end

    let
        split = Split(node.splitpoint, :L)
        _splits = [split; splits]
        _paths!(node.left, _splits, paths)
    end

    let
        split = Split(node.splitpoint, :R)
        _splits = [split; splits]
        _paths!(node.right, _splits, paths)
    end

    return paths
end
