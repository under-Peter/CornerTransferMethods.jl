module CornerTransferMethods
import Base: iterate
using Parameters: @unpack
using Base.Iterators: take, enumerate, rest
using Printf: @printf
using LinearAlgebra: svdvals!, svd!, eigen!, Hermitian, normalize!
using TNTensors
using TensorOperations: @tensor, scalar

export ctm, magnetisation, isingpart
export rotsymCTMState

struct rotsymCTMIterable{T}
    A::DTensor{T,4}
    χ::Int
    Cinit::Union{DTensor{T,2}, Nothing}
    Tinit::Union{DTensor{T,3}, Nothing}
end

function rotsymctmiterable(A::DTensor{T}, χ::Int, Cinit = nothing, Tinit = nothing) where T
    rotsymCTMIterable{T}(A, χ, Cinit, Tinit)
end


mutable struct rotsymCTMState{S}
    C::DTensor{S,2}
    T::DTensor{S,3}
    Cxl::DTensor{S,4}
    Txl::DTensor{S,5}
    oldsvdvals::Vector{S}
    diffs::Vector{S}
    CT::DTensor{S,3}
    CTT::DTensor{S,4}
    CxlZ::DTensor{S,3}
    TxlZ::DTensor{S,4}
    Ctmp::DTensor{S,2}
    Ttmp::DTensor{S,3}
    Cxlmat::DTensor{S,2}
end

function iterate(iter::rotsymCTMIterable{S}) where {S}
    @unpack A, χ = iter
    if iter.Cinit == nothing
        C = initializeC(A, χ)
    else
        C = copy(iter.Cinit)
    end
    if iter.Tinit == nothing
        T = initializeT(A, χ)
    else
        T = copy(iter.Tinit)
    end
    d = size(A,1)

    Cxl  = DTensor(Array{S,4}(undef, χ, d, d, χ))
    Txl  = DTensor(Array{S,5}(undef, χ, d, d, d, χ))
    CT   = DTensor(Array{S,3}(undef, χ, χ, d))
    CTT  = DTensor(Array{S,4}(undef, χ, d, d, χ))
    CxlZ = DTensor(Array{S,3}(undef, χ, d, χ))
    TxlZ = DTensor(Array{S,4}(undef, d, χ, d, χ))
    Ctmp = DTensor(Array{S,2}(undef, χ, χ))
    Ttmp = DTensor(Array{S,3}(undef, χ, d, χ))
    Cxlmat = DTensor(Array{S,2}(undef, χ*d, d*χ))
    diffs = []
    oldsvdvals = zeros(S,χ)
    state = rotsymCTMState{S}(C, T, Cxl, Txl, oldsvdvals, diffs,
                              CT, CTT, CxlZ, TxlZ, Ctmp, Ttmp, Cxlmat)
    return state, state
end

initializeC(A::DTensor{T}, χ) where T = DTensor(rand(T,χ, χ) |> (x -> x + permutedims(x,(2,1))))
initializeT(A::DTensor{T}, χ) where T = DTensor(randn(T,χ, size(A,1), χ) |> (x -> x + permutedims(x,(3,2,1))))

function iterate(iter::rotsymCTMIterable{TT}, state::rotsymCTMState{TT}) where TT
    @unpack A, χ = iter
    @unpack C, T, Cxl, Txl, oldsvdvals, diffs, CT, CTT, CxlZ, TxlZ, Ctmp, Ttmp, Cxlmat = state

    #grow
    @tensor begin
        CT[c2,o1,c4] = C[c1,c2] * T[o1,c4,c1]
        CTT[o1,c4,c3,o4] = CT[c2,o1,c4] * T[c2,c3,o4]
        Cxl[o1,o2,o3,o4] = CTT[o1,c4,c3,o4] * A[o2,o3,c3,c4]
        Txl[o1, o2, o3, o4, o5] = T[o1, c1, o5] * A[o2, o3, o4, c1]
    end
    #renormalize
    Cxlmat, inverter = fuselegs(Cxl, ((1,2),(3,4)))
    U, = tensorsvd(Cxlmat, svdcutfunction = svdcutfun_maxχ(χ))
    Z  = splitlegs(U, ((1,1,1),(1,1,2),2), inverter)

    @tensor CxlZ[o1,c3,c4] = Cxl[c1,c2,c3,c4] * Z[c1,c2,o1]
    @tensor Ctmp[o1,o2]    = CxlZ[o1,c3,c4] * Z[c4,c3,o2]
    @tensor TxlZ[o2,o1,c3,c4] = Txl[c1,c2,o2,c3,c4] * Z[c1,c2,o1]
    @tensor Ttmp[o1,o2,o3]    = TxlZ[o2,o1,c3,c4] * Z[c4,c3,o3]
    #symmetrize
    @tensor begin
        C[o1,o2] = Ctmp[o1,o2] + Ctmp[o2,o1]
        T[o1,o2,o3] = Ttmp[o1,o2,o3] + Ttmp[o3,o2,o1]
    end
    copyto!(Ctmp,C)
    _, S, = tensorsvd(Ctmp)
    vals = diag(S)
    maxval = maximum(vals)
    apply!(C, x -> x ./ maxval)
    apply!(T, x -> x ./ maxval)
    normalize!(vals,1)

    #compare
    push!(diffs, sum(abs, oldsvdvals - vals))
    oldsvdvals[:] = vals
    return state, state
end


ctm(A::T, Asz::T, χ; kwargs...) where {T<:AbstractTensor} =  rotsymctm(A,Asz,χ;kwargs...)

function rotsymctm(A::AbstractTensor{S,4}, Asz::AbstractTensor{S,4}, χ::Int;
                                    Cinit::Union{Nothing, AbstractTensor{S,2}} = nothing,
                                    Tinit::Union{Nothing, AbstractTensor{S,3}} = nothing,
                                    tol::Float64 = 1e-18,
                                    maxit::Int = 5000,
                                    period::Int = 100,
                                    verbose::Bool = true) where S
    stop(state) = length(state.diffs) > 1 && state.diffs[end] < tol
    disp(state) = @printf("%5d \t| %.3e | %.3e | %.3e\n",
                            state[2][1], state[1]/1e9,
                            state[2][2].diffs[end],
                            magnetisation(state[2][2].C,state[2][2].T,A,Asz))

    iter = rotsymctmiterable(A, χ, Cinit, Tinit)
    tol > 0 && (iter = halt(iter, stop))
    iter = take(iter, maxit)
    iter = enumerate(iter)

    if verbose
        @printf("\tn \t| time (s)\t| diff\t\t| mag \n")
        iter = sample(iter, period)
        iter = stopwatch(iter)
        iter = tee(iter, disp)
        (_, (it, state)) = loop(iter)
    else
        (it, state) = loop(iter)
    end
    return  (it, (state.C, state.T))
end

include("auxiliary-iterators.jl")
export atens, asztens
include("ising.jl")

end # module
