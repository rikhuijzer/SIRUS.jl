api_docs = read(joinpath(pkgdir(SIRUS), "docs", "src", "api.md"), String)

# Testing manually because setting doctest too restrictive doesn't work with PlutoStaticHTML.
for name in names(SIRUS)
    @test contains(api_docs, string(name))
end

# warn suppresses warnings when keys already exist.
DocMeta.setdocmeta!(SIRUS, :DocTestSetup, :(using SIRUS); recursive=true, warn=false)
doctest(SIRUS)
