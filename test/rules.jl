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

selected_rules = ST._select_rules(rules; p0=20)
@test eltype(rules) == eltype(selected_rules)
@test length(selected_rules) < length(rules)

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

@test ST._filter_reversed([r1, r2]) == [r1]

r5 = ST.Rule(ST.TreePath(" X[i, 3] < 64 "), [0.56], [0.334])
r7 = ST.Rule(ST.TreePath(" X[i, 1] ≥ 32000 & X[i, 3] ≥ 64 "), [0.517], [0.67])
r12 = ST.Rule(ST.TreePath(" X[i, 1] ≥ 32000 & X[i, 3] < 64 "), [0.192], [0.102])
r15 = ST.Rule(ST.TreePath(" X[i, 1] ≥ 32000 & X[i, 4] < 12 "), [0.192], [0.102])

@test ST._equal_variables_thresholds(r1, r2) == true
@test ST._equal_variables_thresholds(r1, r5) == false
@test ST._equal_variables_thresholds(r5, r12) == false
@test ST._equal_variables_thresholds(r7, r12) == true
@test ST._equal_variables_thresholds(r12, r15) == false

@test ST._gap_width(r12) < ST._gap_width(r7)
@test ST._filter_linearly_dependent([r1, r5, r7, r12]) == [r1, r5, r7]

# The other two examples, I don't understand.
# Either I'm wrong or the examples are wrong. CHMIN < 12 is not a stand-alone rule.

@test ST._predict([r1], [31000]) == [0.61]
@test ST._predict([r1], [33000]) == [0.408]
@test ST._predict([r1, r5], [33000, 0, 61]) == [mean([0.408, 0.56])]

