text = " X[i, 1] < 1.0 & X[i, 1] ≥ 4.0 "
@test repr(TreePath(text)) == "TreePath(\"$text\")"

Float = ST.Float
classes = [:a, :b, :c]
left = ST.Leaf([1.0, 0.0, 0.0])
splitpoint = ST.SplitPoint(1, Float(1))
right = ST.Node(splitpoint, ST.Leaf([0.0, 1.0, 0.0]), ST.Leaf([0.0, 0.0, 1.0]))

splitpoint = ST.SplitPoint(1, ST.Float(4))
node = ST.Node(splitpoint, left, right)

paths = ST._rules!(node)

n = 200
rng = StableRNG(1)
X, y = make_moons(n; rng, shuffle=true)
model = StableForestClassifier(; rng)
mach = machine(model, X, y)
fit!(mach; verbosity=0)
forest = mach.fitresult

# paths = ST._paths(forest)
# selected_rules = ST._select_rules(paths; p0=20)
# @test eltype(paths) == eltype(selected_rules)
# @test length(selected_rules) < length(paths)

# These rules are roughly equal to the ones in Table 3 of the Supplementary PDF
# (https://proceedings.mlr.press/v130/benard21a.html).

# Let the features be
# 1: MMAX
# 2: MMIN
# 3: CACH
# 4: CHMIN
# 5: MYCT
r1 = ST.Rule(ST.TreePath(" X[i, 1] < 32000 "), [0.61], [0.408])
r2 = ST.Rule(ST.TreePath(" X[i, 1] ≥ 32000 "), [0.408], [0.61])


