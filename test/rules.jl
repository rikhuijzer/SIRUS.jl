text = " X[i, 1] < 1.0 & X[i, 1] ≥ 4.0 "
@test repr(TreePath(text)) == "TreePath(\"$text\")"

Float = ST.Float
classes = [:a, :b, :c]
left = ST.Leaf([1.0, 0.0, 0.0])
splitpoint = ST.SplitPoint(1, Float(1))
right = ST.Node(splitpoint, ST.Leaf([0.0, 1.0, 0.0]), ST.Leaf([0.0, 0.0, 1.0]))

splitpoint = ST.SplitPoint(1, ST.Float(4))
node = ST.Node(splitpoint, left, right)

rules = ST._rules!(node)

n = 200
rng = StableRNG(1)
X, y = make_moons(n; rng, shuffle=true)
model = StableForestClassifier(; rng)
mach = machine(model, X, y)
fit!(mach; verbosity=0)
forest = mach.fitresult

rules = ST._rules(forest)

r1 = ST.Rule(ST.TreePath(" X[i, 1] < 32000 "), [0.61], [0.408])
r1b = ST.Rule(ST.TreePath(" X[i, 1] < 32000 "), [0.61], [0.408])

@test hash(r1) == hash(r1b)
@test ST._count_unique([1, 1, 1, 2]) == Dict(1 => 3, 2 => 1)

selected_rules = ST._select_rules(rules; p0=0.001)
@test eltype(rules) == eltype(selected_rules)
# Ouch. It isn't stable.
@test 140 < length(selected_rules) < 300
@test length(ST._filter_linearly_dependent(selected_rules)) < 40

r5 = ST.Rule(ST.TreePath(" X[i, 3] < 64 "), [0.56], [0.334])

@test ST._predict([r1], [31000]) == [0.61]
@test ST._predict([r1], [33000]) == [0.408]
@test ST._predict([r1, r5], [33000, 0, 61]) == [mean([0.408, 0.56])]

