using RCall

@rlibrary sirus

r = R"rnorm(10)"

@show r

R"sessionInfo()"
