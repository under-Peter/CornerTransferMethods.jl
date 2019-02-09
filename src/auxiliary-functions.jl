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

function transfermat(t::AbstractTensor{T,3}, pop::AbstractTensor{T,2}) where T
    @tensor tt[1,2,3,4] := t[1,-1,3] * pop[-1,-2] * t'[2,-2,4]
    tmat = fuselegs(tt,((1,2),(3,4)),InOut(1,-1))[1]
    return tmat
end
#
function clength(t::AbstractTensor{T,3}) where T
    tfun = transferop(t)
    v0 = checked_similar_from_indices(nothing, T, (1,3),(),t,:Y)
    initwithrand!(v0)
    e1, e2 = eigsolve(tfun, v0, 2, :LR, ishermitian = true)[1][1:2]
    return 1/log(2,e1/e2)
end

const ξoft = clength

#
function transferop(t::AbstractTensor{<:Any,3})
    let t = t
        x -> @tensor tx[1,2] := t[1,-3,-1] * x[-1,-2] * t'[2,-3,-2]
    end
end

function transferop(t::AbstractTensor{<:Any,3}, pop::AbstractTensor{<:Any,2})
    let t = t, pop = pop
        x -> @tensor tx[1,2] := t[1,-3,-1] * pop[-3,-4] * x[-1,-2] * t'[2,-4,-2]
    end
end

function transferopevals(t::AbstractTensor{T,3}, n::Int, pop::Union{AbstractTensor{T,2},Nothing} = nothing) where T
    tfun = pop == nothing ? transferop(t) : transferop(t,pop)
    v0 = checked_similar_from_indices(nothing, T, (1,3),(),t,:Y)
    initwithrand!(v0)
    return eigsolve(tfun, v0, n, :LR, ishermitian = true)[1][1:n]
end

function transferopevals(t::DASTensor{T,3}, n::Int, pop::Union{DASTensor{T,2},Nothing} = nothing; ch::DASCharge = zero(chargetype(t))) where T
    tfun = pop == nothing ? transferop(t) : transferop(t,pop)
    v0 = checked_similar_from_indices(nothing, T, (1,3),(),t,:Y)
    setcharge!(v0, ch)
    initwithrand!(v0)
    return eigsolve(tfun, v0, n, :LR, ishermitian = true)[1][1:n]
end

const ξsoft = transferopevals
