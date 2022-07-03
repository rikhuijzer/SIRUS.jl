### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 7c10c275-54d8-4f1a-947f-7861199cdf21
# ╠═╡ show_logs = false
# hideall
begin
	PKGDIR = dirname(dirname(@__DIR__))
	PROJECT_DIR = @__DIR__
	using Pkg: Pkg
	Pkg.activate(PROJECT_DIR)
	Pkg.develop(; path=PKGDIR)
end

# ╔═╡ 148bdc38-19e8-4dfc-80d5-ffeaee28b804
using StableRNGs

# ╔═╡ 5366a8f7-1465-4f0a-b9c8-096818104c24
using CategoricalArrays: categorical

# ╔═╡ cc535dcf-dd60-4fa3-bfdc-90c47438f3fc
using CairoMakie # hideall

# ╔═╡ 758d8562-f88d-47bd-82aa-22caeda9c208
using StableTrees

# ╔═╡ aa560aad-9de4-4e7f-92ce-316f88439d57
md"""
This package implements the **S**table and **I**nterpretable **RU**le **S**ets (SIRUS) for classification an algorithm by Bénard et al., ([2021](https://doi.org/10.1214/20-EJS1792)).
Regression is also technically possible but not yet implemented.

## Algorithm

Decision tree-based algorithms are known to be unstable.
In other words, the trees can change drastically for small changes in the data.
This is caused by the process in which the trees choose their splits.
To choose the splits, a greedy recursive binary splitting algorithm is used (James et al., [2014](https://doi.org/10.1007/978-1-0716-1418-1)).
For example, consider the following two-dimensional case:
"""

# ╔═╡ e7861f63-aa29-419d-a458-275c8ca9bcfb
n = 10

# ╔═╡ 679abca3-9f22-4a43-a4b5-77dbea63bf08
X = rand(StableRNG(1), n, 2);

# ╔═╡ 73666382-d78e-4096-a97b-7ed90b88d694
y = categorical(rand(StableRNG(1), 0:1, n));

# ╔═╡ 83700ef9-f833-49d0-9ee9-76eb56f643e9
# hideall
markers(Y) = [y == 1 ? :circle : :cross for y in Y]

# ╔═╡ 2dcd43e6-41b9-412b-b5bc-550a89376497
# hideall
let
	fig = Figure()
	ax = Axis(fig[1, 1]; xlabel="X[:, 1]", ylabel="X[:, 2]")
	scatter!(ax, X[:, 1], X[:, 2], markersize=14, marker=markers(y))
	fig
end

# ╔═╡ 0ca8bb9a-aac1-41a7-b43d-314a4029c205
ST = StableTrees;

# ╔═╡ 0e0252e7-87a8-49e4-9a48-5612e0ded41b


# ╔═╡ Cell order:
# ╠═7c10c275-54d8-4f1a-947f-7861199cdf21
# ╠═aa560aad-9de4-4e7f-92ce-316f88439d57
# ╠═148bdc38-19e8-4dfc-80d5-ffeaee28b804
# ╠═e7861f63-aa29-419d-a458-275c8ca9bcfb
# ╠═679abca3-9f22-4a43-a4b5-77dbea63bf08
# ╠═5366a8f7-1465-4f0a-b9c8-096818104c24
# ╠═73666382-d78e-4096-a97b-7ed90b88d694
# ╠═cc535dcf-dd60-4fa3-bfdc-90c47438f3fc
# ╠═83700ef9-f833-49d0-9ee9-76eb56f643e9
# ╠═2dcd43e6-41b9-412b-b5bc-550a89376497
# ╠═758d8562-f88d-47bd-82aa-22caeda9c208
# ╠═0ca8bb9a-aac1-41a7-b43d-314a4029c205
# ╠═0e0252e7-87a8-49e4-9a48-5612e0ded41b
