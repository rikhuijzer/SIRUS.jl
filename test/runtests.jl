using ExplainableRules
using Test

const ER = ExplainableRules
const Float = ExplainableRules.Float

@test ER.gini([1, 1, 1], [1]) == 0.0f0

feature = 1
@test collect(ER._view_y([1 2; 3 4], [1 2], feature, <, 2)) == [1]
@test collect(ER._view_y([1 2; 3 4], [1 2], feature, >, 2)) == [2]

@test ER._find_split([1 2; 3 4], vec([1 2]), vec([1 2])) == (1, 3)
