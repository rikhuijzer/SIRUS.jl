# Serve docs with
# ```
# $ julia -i docs/serve.jl
# ```
using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=dirname(@__DIR__))
using LiveServer
using SIRUS

LiveServer.servedocs()
