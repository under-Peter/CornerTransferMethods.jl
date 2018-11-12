using LinearAlgebra: eigvals, normalize
cornereoe(c::AbstractTensor{T,2}) where T = cornereoe(toarray(c))

function cornereoe(c::Array{T,2}) where T
    ev4 = normalize(eigvals(c),4).^4
    return -sum(ev4 .* log.(2,ev4))
end
