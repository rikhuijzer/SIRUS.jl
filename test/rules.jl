text = " X[i, 1] < 1.0 & X[i, 1] ≥ 4.0 "
@test repr(TreePath(text)) == "TreePath(\"$text\")"

l = 1
T = Int
left = ST.Leaf{T}(1, l)
splitpoint = ST.SplitPoint(1, ST.Float(1))
right = ST.Node{T}(splitpoint, ST.Leaf{T}(2, l), ST.Leaf{T}(3, l))

splitpoint = ST.SplitPoint(1, ST.Float(4))
node = ST.Node{T}(splitpoint, left, right)

paths = Set(ST._paths!(node))

