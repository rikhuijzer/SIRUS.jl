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

n = 200
rng = StableRNG(1)
X, y = make_moons(n; rng, shuffle=true)
model = StableForestClassifier(; rng)
mach = machine(model, X, y)
fit!(mach)
forest = mach.fitresult


paths = ST._paths(forest)
selected_rules = ST._select_rules(paths; p0=20)
@test eltype(paths) == eltype(selected_rules)
@test length(selected_rules) < length(paths)
