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
    Path

A path of length `d` is defined as consisting of `d` splits.
See SIRUS paper page 434.
Typically, `d ≤ 2`.
Note that a path can also be a path to a node; not necessarily a leaf.
"""
struct Path
    splits::Vector{Split}
end

function Path(text::String)
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
    return Path(splits)
end

function show(io::IO, path::Path)
    texts = map(path.splits) do split
        splitpoint = split.splitpoint
        comparison = split.direction == :L ? '<' : '≥'
        text = "X[i, $(splitpoint.feature)] $comparison $(splitpoint.value)"
    end
    text = string("Path(\" ", join(texts, " & "), " \")")::String
    print(io, text)
end

function _paths!(leaf::Leaf, splits::Vector{Split}, paths::Vector{Path})
    path = Path(splits)
    push!(paths, path)
    return nothing
end

"""
Return a set of all the paths in the `tree`.
This is the rule generation step of SIRUS.
"""
function _paths!(node::Node, splits::Vector{Split}=Split[], paths::Vector{Path}=Path[])
    if !isempty(splits)
        path = Path(splits)
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
