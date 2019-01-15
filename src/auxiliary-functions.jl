using LinearAlgebra: eigvals, normalize
using Arpack: eigs

cornereoe(c::AbstractTensor{T,2}) where T = cornereoe(toarray(c))

function cornereoe(c::Array{T,2}) where T
    ev4 = normalize(eigvals(c),4).^4
    return -sum(ev4 .* log.(2,ev4))
end

function transfermat(t::AbstractTensor{T,3}) where T
    @tensor tt[1,2,3,4] := t[1,-1,3] * t'[2,-1,4]
    tmat = fuselegs(tt,((1,2),(3,4)))[1]
    return tmat
end

function clength(t::AbstractTensor{T,3}) where T
    evs = eigs(transfermat(toarray(t)), nev=2)[1]
    return 1/log(2,evs[1]/evs[2])
end
