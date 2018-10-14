module CornerTransferMethods
import Base: iterate
using TensorOperations: @tensor, scalar
using Parameters: @unpack
using Base.Iterators: take, enumerate, rest
using Printf: @printf
using LinearAlgebra: svdvals!, svd!, eigen!, Hermitian

export ctm, magnetisation, isingpart
export rotsymCTMState

struct rotsymCTMIterable{S}
    A::Array{S,4}
    χ::Int
    d::Int
    Cinit::Union{Array{S,2}, Nothing}
    Tinit::Union{Array{S,3}, Nothing}
end

function rotsymctmiterable(A::Array{T,4}, χ::Int, Cinit = nothing, Tinit = nothing) where T
    rotsymCTMIterable{T}(A, χ, size(A,1), Cinit, Tinit)
end


mutable struct rotsymCTMState{S}
    C::Array{S,2}
    T::Array{S,3}
    Cxl::Array{S,4}
    Txl::Array{S,5}
    oldsvdvals::Vector{S}
    diffs::Vector{S}
    CT::Array{S,3}
    CTT::Array{S,4}
    CxlZ::Array{S,3}
    TxlZ::Array{S,4}
    Ctmp::Array{S,2}
    Ttmp::Array{S,3}
    Cxlmat::Array{S,2}
end

function iterate(iter::rotsymCTMIterable{S}) where {S}
    @unpack A, d, χ = iter
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

    Cxl  = Array{S,4}(undef, χ, d, d, χ)
    Txl  = Array{S,5}(undef, χ, d, d, d, χ)
    CT   = Array{S,3}(undef, χ, χ, d)
    CTT  = Array{S,4}(undef, χ, d, d, χ)
    CxlZ = Array{S,3}(undef, χ, d, χ)
    TxlZ = Array{S,4}(undef, d, χ, d, χ)
    Ctmp = Array{S,2}(undef, χ, χ)
    Ttmp = Array{S,3}(undef, χ, d, χ)
    Cxlmat = Array{S,2}(undef, χ*d, d*χ)
    diffs = []
    oldsvdvals = zeros(χ)
    state = rotsymCTMState{S}(C, T, Cxl, Txl, oldsvdvals, diffs,
                              CT, CTT, CxlZ, TxlZ, Ctmp, Ttmp, Cxlmat)
    return state, state
end

initializeC(A::Array{T}, χ) where T = rand(T,χ, χ) |> (x -> x + permutedims(x,(2,1)))
initializeT(A::Array{T}, χ) where T = randn(T,χ, size(A,1), χ) |> (x -> x + permutedims(x,(3,2,1)))

function iterate(iter::rotsymCTMIterable{S}, state::rotsymCTMState{S}) where S
    @unpack A, d, χ = iter
    @unpack C, T, Cxl, Txl, oldsvdvals, diffs, CT, CTT, CxlZ, TxlZ, Ctmp, Ttmp, Cxlmat = state

    #grow
    @tensor begin
        # Cxl[o1, o2, o3, o4] = C[c1,c2] * T[o1,c4,c1] * T[c2,c3,o4] * A[o2,o3,c3,c4]
        CT[c2,o1,c4] = C[c1,c2] * T[o1,c4,c1]
        CTT[o1,c4,c3,o4] = CT[c2,o1,c4] * T[c2,c3,o4]
        Cxl[o1,o2,o3,o4] = CTT[o1,c4,c3,o4] * A[o2,o3,c3,c4]
        Txl[o1, o2, o3, o4, o5] = T[o1, c1, o5] * A[o2, o3, o4, c1]
    end
    #renormalize
    Cxlmat[:] = reshape(Cxl, χ*d, d*χ) * reshape(Cxl, χ*d, d*χ)'
    U = eigen!(Hermitian(Cxlmat)).vectors[:,end:-1:end-χ+1]

    Z = reshape(U, χ, d, χ)
    @tensor begin
        # C[o1, o2] = Cxl[c1,c2,c3,c4] * Z[c1,c2,o1] * Z[c4,c3,o2]
        CxlZ[o1,c3,c4] = Cxl[c1,c2,c3,c4] * Z[c1,c2,o1]
        Ctmp[o1,o2]    = CxlZ[o1,c3,c4] * Z[c4,c3,o2]
        # T[o1, o2, o3] = Txl[c1,c2,o2,c3,c4] * Z[c1,c2,o1] * Z[c4,c3,o3]
        TxlZ[o2,o1,c3,c4] = Txl[c1,c2,o2,c3,c4] * Z[c1,c2,o1]
        Ttmp[o1,o2,o3]    = TxlZ[o2,o1,c3,c4] * Z[c4,c3,o3]
    end
    #symmetrize
    C[:] = Ctmp
    C .+= permutedims(Ctmp)
    Ctmp[:] = C
    T[:] = Ttmp
    T .+= permutedims(Ttmp, (3,2,1))
    vals = svdvals!(Ctmp)[1:χ]
    C    ./= vals[1]
    T    ./= vals[1]
    vals ./= vals[1]

    #compare
    push!(diffs, sum(abs, oldsvdvals - vals))
    oldsvdvals[:] = vals
    return state, state
end

function ctm(A::Array{S,4}, Asz::Array{S,4}, χ::Int;
                                    Cinit::Union{Nothing, Array{S,2}} = nothing,
                                    Tinit::Union{Nothing, Array{S,3}} = nothing,
                                    tol::Float64 = 1e-18,
                                    maxit::Int = 5000,
                                    period::Int = 100,
                                    verbose::Bool = true) where S
    stop(state) = length(state.diffs) > 1 && state.diffs[end] < tol
    disp(state) = @printf("%5d \t| %.3e | %.3e | %.3e\n", state[2][1], state[1]/1e9,
                            state[2][2].diffs[end], magnetisation(state[2][2].C,state[2][2].T,A,Asz))
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
