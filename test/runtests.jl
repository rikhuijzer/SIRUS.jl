using ExplainableRules
using Test

const ER = ExplainableRules
const Float = ExplainableRules.Float

@test ER.gini([1, 1, 1], [1]) == 0.0f0
