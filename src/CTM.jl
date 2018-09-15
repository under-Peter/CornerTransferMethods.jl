module CTM
import Base: iterate
using TensorOperations: @tensor
using Parameters: @unpack
# using KrylovKit: svdsolve
using Base.Iterators: take, enumerate, rest
using Printf: @printf
using LinearAlgebra: tr, svd, svdvals


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

# """
# rotsymCTMState contains five tensors:
#      C:             Cxl
#     +-+ χ           +-+  +-+
#     |C+--+2         |C+--+T+--+4
#     +++             +++  +++ χ
#      |χ              |    |
#      +1             +++  +++
#                     |T+--+A+--+3
#                     +++  +++ d
#                      |χ    |d
#                      +1   +2
#
#     T:               Txl:
#      +3              +5   +4
#      |χ              |χ   |d
#     +++             +++  +++
#     |T+--+2         |T+--+A+--+3
#     +++ d           +++  +++ d
#      |χ              |χ   |d
#      +1              +1   +2
#
#      Z:
#       +3
#        |χ
#       / \
#      / Z \
#      +---+
#      |χ  |d
#      +1  +2
# """

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

    Cxl   = Array{S,4}(undef, χ, d, d, χ)
    Txl   = Array{S,5}(undef, χ, d, d, d, χ)
    CT = Array{S,3}(undef, χ, χ, d)
    CTT  = Array{S,4}(undef, χ, d, d, χ)
    CxlZ = Array{S,3}(undef, χ, d, χ)
    TxlZ = Array{S,4}(undef, d, χ, d, χ)
    Ctmp = Array{S,2}(undef, χ, χ)
    Ttmp = Array{S,3}(undef, χ, d, χ)
    diffs = []
    oldsvdvals = zeros(χ)
    state = rotsymCTMState{S}(C, T, Cxl, Txl, oldsvdvals, diffs,
                              CT, CTT, CxlZ, TxlZ, Ctmp, Ttmp)
    return state, state
end

initializeC(A, χ) = randn(χ, χ) |> (x -> x + permutedims(x,(2,1)))
initializeT(A, χ) = randn(χ, size(A,1), χ) |> (x -> x + permutedims(x,(3,2,1)))

function iterate(iter::rotsymCTMIterable{S}, state::rotsymCTMState{S}) where S
    @unpack A, d, χ = iter
    @unpack C, T, Cxl, Txl, oldsvdvals, diffs, CT, CTT, CxlZ, TxlZ, Ctmp, Ttmp = state

    #grow
    @tensor begin
        # Cxl[o1, o2, o3, o4] = C[c1,c2] * T[o1,c4,c1] * T[c2,c3,o4] * A[o2,o3,c3,c4]
        CT[c2,o1,c4] = C[c1,c2] * T[o1,c4,c1]
        CTT[o1,c4,c3,o4] = CT[c2,o1,c4] * T[c2,c3,o4]
        Cxl[o1,o2,o3,o4] = CTT[o1,c4,c3,o4] * A[o2,o3,c3,c4]
        Txl[o1, o2, o3, o4, o5] = T[o1, c1, o5] * A[o2, o3, o4, c1]
    end
    #renormalize
    # vals, U, V, info  = svdsolve(reshape(Cxl, χ*d, d*χ), χ, krylovdim = 32)
    U = svd(reshape(Cxl, χ*d, d*χ)).U[:,1:χ]
    # @assert info.converged > χ "svd did not converge"
    # Z = reshape(hcat(U[1:χ]...), χ, d, χ)
    Z = reshape(U, χ, d, χ)
    @tensor begin
        # C[o1, o2] = Cxl[c1,c2,c3,c4] * Z[c1,c2,o1] * Z[c4,c3,o2]
        # T[o1, o2, o3] = Txl[c1,c2,o2,c3,c4] * Z[c1,c2,o1] * Z[c4,c3,o3]
        CxlZ[o1,c3,c4] = Cxl[c1,c2,c3,c4] * Z[c1,c2,o1]
        Ctmp[o1,o2]    = CxlZ[o1,c3,c4] * Z[c4,c3,o2]
        TxlZ[o2,o1,c3,c4] = Txl[c1,c2,o2,c3,c4] * Z[c1,c2,o1]
        Ttmp[o1,o2,o3]    = TxlZ[o2,o1,c3,c4] * Z[c4,c3,o3]
    end
    #symmetrize
    C[:] = Ctmp
    C .+= permutedims(Ctmp,(2,1))
    T[:] = Ttmp
    T .+= permutedims(Ttmp, (3,2,1))
    # vals, = svdsolve(C, χ, krylovdim = 32)
    # vals = vals[1:χ]
    vals = svdvals(C)[1:χ]
    C    ./= vals[1]
    T    ./= vals[1]
    vals ./= vals[1]

    #compare
    push!(diffs, sum(x -> x^2, oldsvdvals - vals))
    oldsvdvals[:] = vals
    return state, state
end

function ctm(A::Array{S,4}, χ::Int; Cinit::Union{Nothing, Array{S,2}} = nothing,
                                    Tinit::Union{Nothing, Array{S,3}} = nothing,
                                    tol::Float64 = 1e-9,
                                    maxit::Int = 100,
                                    period::Int = 10,
                                    verbose::Bool = true) where S
    stop(state) = length(state.diffs) > 1 && state.diffs[end] < tol
    disp(state) = @printf("%5d \t| %.3e | %.3e | %.3e\n", state[2][1], state[1]/1e9,
                            state[2][2].diffs[end], magnetisation(state[2][2]))
    iter = rotsymctmiterable(A, χ, Cinit, Tinit)
    iter = halt(iter, stop)
    iter = take(iter, maxit)
    iter = enumerate(iter)

    if verbose
        @printf("\tn \t| time (ns)\t| diff\t\t| mag \n")
        iter = sample(iter, period)
        iter = stopwatch(iter)
        iter = tee(iter, disp)
        (_, (it, state)) = loop(iter)
    else
        (it, state) = loop(iter)
    end
    return  (it, state)
end

#= iterators from https://lostella.github.io/blog/2018/07/25/iterative-methods-done-right =#
#Halting
struct HaltingIterable{I, F}
    iter::I
    fun::F
end

function iterate(iter::HaltingIterable)
    next = iterate(iter.iter)
    return dispatch(iter, next)
end

function iterate(iter::HaltingIterable, (instruction, state))
    if instruction == :halt return nothing end
    next = iterate(iter.iter, state)
    return dispatch(iter, next)
end

function dispatch(iter::HaltingIterable, next)
    if next === nothing return nothing end
    return next[1], (iter.fun(next[1]) ? :halt : :continue, next[2])
end

halt(iter::I, fun::F) where {I, F} = HaltingIterable{I, F}(iter, fun)

#Tee
struct TeeIterable{I, F}
    iter::I
    fun::F
end

function iterate(iter::TeeIterable, args...)
    next = iterate(iter.iter, args...)
    if next !== nothing iter.fun(next[1]) end
    return next
end

tee(iter::I, fun::F) where {I, F} = TeeIterable{I, F}(iter, fun)

#Sampling
struct SamplingIterable{I}
    iter::I
    period::UInt
end

function iterate(iter::SamplingIterable, state=iter.iter)
    current = iterate(state)
    if current === nothing return nothing end
    for i = 1:iter.period-1
        next = iterate(state, current[2])
        if next === nothing return current[1], rest(state, current[2]) end
        current = next
    end
    return current[1], rest(state, current[2])
end

sample(iter::I, period) where I = SamplingIterable{I}(iter, period)

#Timing
struct StopwatchIterable{I}
    iter::I
end

function iterate(iter::StopwatchIterable)
    t0 = time_ns()
    next = iterate(iter.iter)
    return dispatch(iter, t0, next)
end

function iterate(iter::StopwatchIterable, (t0, state))
    next = iterate(iter.iter, state)
    return dispatch(iter, t0, next)
end

function dispatch(iter::StopwatchIterable, t0, next)
    if next === nothing return nothing end
    return (time_ns()-t0, next[1]), (t0, next[2])
end

stopwatch(iter::I) where I = StopwatchIterable{I}(iter)

#loop
function loop(iter)
    x = nothing
    for y in iter x = y end
    return x
end

#ising-tensor
function partitionfun(h, β)
    tensor = Array{Float64, 4}(undef, 2,2,2,2)
    for i=1:2, j=1:2, k=1:2, l=1:2
        tensor[i,j,k,l] = exp(-β * h(i, j, k, l))
    end
    return tensor
end

function ising(i, j, k, l)
    spin = (1, -1)
    return sum(map((x,y) -> -spin[x]*spin[y], (i,j,k,l), (l,i,j,k)))
end

function ising(mag, i, j, k, l)
    spin = (1, -1)
    e = ising(i,j,k,l)
    e += mag/2 * sum(x->spin[x], [i,j,k,l])
    return e
end

ising(mag) = (x...) -> ising(mag, x...)

isingpart(β) = partitionfun(ising, β)

function magnetisation(state::rotsymCTMState)
    sz = [1 0; 0 -1]
    @unpack C,T = state
    @tensor o[o1,o2] := C[c1,c2] * C[c2,c3] * T[c5,o1,c3] *
                        C[c5,c7] * C[c7,c8] * T[c8,o2,c1];
    return tr(sz * o)/ tr(o)
end


end # module
