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

struct rotsymCTMIterable{T, TA <: AbstractTensor{T,4},
                            TC <: AbstractTensor{T,2},
                            TT <: AbstractTensor{T,3}}
    A::TA
    χ::Int
    Cinit::Union{TC, Nothing}
    Tinit::Union{TT, Nothing}
end

function rotsymctmiterable(A::AbstractTensor{T}, χ::Int, Cinit = nothing, Tinit = nothing) where T
    tnt = typeof(A).name.wrapper
    TC = tnt{T,2}
    TT = tnt{T,3}
    TA = tnt{T,4}
    rotsymCTMIterable{T,TA,TC,TT}(A, χ, Cinit, Tinit)
end

function rotsymctmiterable(A::ZNTensor{T,N,M}, χ::Int, Cinit = nothing, Tinit = nothing) where {T,N,M}
    TC = ZNTensor{T,2,M}
    TT = ZNTensor{T,3,M}
    TA = ZNTensor{T,4,M}
    rotsymCTMIterable{T,TA,TC,TT}(A, χ, Cinit, Tinit)
end

#mutable
struct rotsymCTMState{S, TS2 <: AbstractTensor,
                                TS3 <: AbstractTensor,
                                TS4 <: AbstractTensor,
                                TS5 <: AbstractTensor}
    C::TS2
    T::TS3
    oldsvdvals::Vector{S}
    diffs::Vector{S}
    Cxl::TS4
    Txl::TS5
    CT::TS3
    CTT::TS4
    CxlZ::TS3
    TxlZ::TS4
    Ctmp::TS2
    Ttmp::TS3
end

_znhelper(a::ZNTensor{T,N,M}) where {T,N,M} = M

function iterate(iter::rotsymCTMIterable{S,TA,TC,TT}) where {S,TA,TC,TT}
    @unpack A, χ, Cinit, Tinit = iter
    if Cinit == nothing
        C = initializeC(A, χ)
    else
        C = copy(Cinit)
    end
    if Tinit == nothing
        T = initializeT(A, χ)
    else
        T = copy(Tinit)
    end

    if TA <: ZNTensor
        T5 = ZNTensor{S,5,_znhelper(A)}
    else
        T5 = typeof(A).name.wrapper{S,5}
    end

    caches = initializeCaches(A, χ)
    l = length(diag(C))
    oldsvdvals = zeros(S,l)
    state = rotsymCTMState{S,TC,TT,TA,T5}(C, T, oldsvdvals, [], caches...)
    return state, state
end

function initializeCaches(A::DTensor{S,4}, χ) where {S}
    d = size(A,1)
    Cxl  = DTensor{S,4}((χ, d, d, χ))
    Txl  = DTensor{S,5}((χ, d, d, d, χ))
    CT   = DTensor{S,3}((χ, χ, d))
    CTT  = DTensor{S,4}((χ, d, d, χ))
    CxlZ = DTensor{S,3}((χ, d, χ))
    TxlZ = DTensor{S,4}((d, χ, d, χ), )
    Ctmp = DTensor{S,2}((χ, χ))
    Ttmp = DTensor{S,3}((χ, d, χ))
    return (Cxl, Txl, CT, CTT, CxlZ, TxlZ, Ctmp, Ttmp)
end
initializeC(A::DTensor{T}, χ) where T = DTensor(rand(T,χ, χ) |> (x -> x + permutedims(x,(2,1))))
initializeT(A::DTensor{T}, χ) where T = DTensor(randn(T,χ, size(A,1), χ) |> (x -> x + permutedims(x,(3,2,1))))

function initializeCaches(A::ZNTensor{S,4,N}, χ) where {S,N}
    ds = sizes(A,1)
    χs = [χ for i in 1:N]
    Cxl  = ZNTensor{S,4,N}((χs, ds, ds, χs),(1,1,1,1))
    Txl  = ZNTensor{S,5,N}((χs, ds, ds, ds, χs),(1,1,1,1,1))
    CT   = ZNTensor{S,3,N}((χs, χs, ds),(1,1,-1))
    CTT  = ZNTensor{S,4,N}((χs, ds, ds, χs),(1,1,1,1))
    CxlZ = ZNTensor{S,3,N}((χs, ds, χs),(1,1,1))
    TxlZ = ZNTensor{S,4,N}((ds, χs, ds, χs), (1,-1,1,1))
    Ctmp = ZNTensor{S,2,N}((χs, χs),(1,1))
    Ttmp = ZNTensor{S,3,N}((χs, ds, χs),(-1,1,-1))
    return (Cxl, Txl, CT, CTT, CxlZ, TxlZ, Ctmp, Ttmp)
end

function initializeC(A::ZNTensor{T,4,N}, χ) where {T,N}
    χs = [χ for i in 1:N]
    C1 = rand(ZNTensor{T,2,N}, (χs,χs), (-1,-1))
    @tensor C2[1,2] := C1[1,2] + C1[2,1]
    return C2
 end

function initializeT(A::ZNTensor{T,4,N}, χ) where {T,N}
    ds = sizes(A,1)
    χs = [χ for i in 1:N]
    T1 = rand(ZNTensor{T,3,N}, (χs,ds,χs), (1,-1,1))
    @tensor T2[1,2,3] := T1[1,2,3] + T1[3,2,1]
    return T2
 end

function iterate(iter::rotsymCTMIterable, state::rotsymCTMState)
    @unpack A, χ = iter
    @unpack C, T, Cxl, Txl, oldsvdvals, diffs, CT, CTT, CxlZ, TxlZ, Ctmp, Ttmp = state

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

    @tensor begin
        CxlZ[o1,c3,c4] = Cxl[c1,c2,c3,c4] * Z[c1,c2,o1]
        Ctmp[o1,o2]    = CxlZ[o1,c3,c4] * Z[c4,c3,o2]
        TxlZ[o2,o1,c3,c4] = Txl[c1,c2,o2,c3,c4] * Z'[c1,c2,o1]
        Ttmp[o1,o2,o3]    = TxlZ[o2,o1,c3,c4] * Z'[c4,c3,o3]
    end
    #symmetrize
    @tensor begin
        C[o1,o2] = Ctmp[o1,o2] + Ctmp[o2,o1]
        T[o1,o2,o3] = Ttmp[o1,o2,o3] + Ttmp[o3,o2,o1]
    end
    copyto!(Ctmp,C)
    _, s, = tensorsvd(Ctmp)
    vals = sort!(diag(s))
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
export atens, asztens, atenses, atensesz2
include("ising.jl")

end # module
