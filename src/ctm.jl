include("auxiliary-iterators.jl")
struct rotsymCTMIterable{T, TA <: AbstractTensor{T,4}, TC <: AbstractTensor{T,2}, TT <: AbstractTensor{T,3}}
    A::TA
    χ::Int
    Cinit::Union{TC, Nothing}
    Tinit::Union{TT, Nothing}
end

function rotsymctmiterable(A::DTensor{T,4}, χ::Int,
        Cinit = nothing, Tinit = nothing) where T
    rotsymCTMIterable{T,DTensor{T,4},DTensor{T,2},DTensor{T,3}}(A, χ, Cinit, Tinit)
end

function rotsymctmiterable(A::DASTensor{T,4,SYM,CHS,SS,CH}, χ::Int,
        Cinit = nothing, Tinit = nothing) where {T,N,SYM,CHS,SS,CH}
    TC = DASTensor{T,2,SYM,CHS,SS,CH}
    TT = DASTensor{T,3,SYM,CHS,SS,CH}
    TA = DASTensor{T,4,SYM,CHS,SS,CH}
    rotsymCTMIterable{T,TA,TC,TT}(A, χ, Cinit, Tinit)
end

#mutable
struct rotsymCTMState{S, TA <: AbstractTensor, TC <: AbstractTensor, TT <: AbstractTensor}
    C::TC
    T::TT
    oldsvdvals::Vector{S}
    diffs::Vector{S}
end


function iterate(iter::rotsymCTMIterable{S,TA,TC,TT}) where {S,TA,TC,TT}
    @unpack A, χ, Cinit, Tinit = iter
    C = Cinit == nothing ? initializeC(A,χ) : deepcopy(Cinit)
    T = Tinit == nothing ? initializeT(A,χ) : deepcopy(Tinit)

    l = ifelse(C isa DASTensor, 2χ, χ)
    oldsvdvals = zeros(S,l)
    state = rotsymCTMState{S,TA,TC,TT}(C, T, oldsvdvals, [])
    return state, state
end

initializeC(A::DTensor{T}, χ) where T =
    DTensor(rand(T,χ, χ) |> (x -> x + permutedims(x,(2,1))))
initializeT(A::DTensor{T}, χ) where T =
    DTensor(randn(T,χ, size(A,1), χ) |> (x -> x + permutedims(x,(3,2,1))))

function initializeC(A::DASTensor{T,4}, χ) where T
    C = checked_similar_from_indices(nothing, T, (1,2), A, :Y)
    s1, s2 = sizes(C)
    s1 .= χ
    s2 .= χ
    initwithrand!(C)
    @tensor C[1,2] := C[1,2] + C[2,1]
    return C
 end

function initializeT(A::DASTensor{S,4}, χ) where S
    T = checked_similar_from_indices(nothing, S, (1,2), (1,), (1,3,2), A, A, :N, :Y)
    s1, s2, s3 = sizes(T)
    s1 .= χ
    s3 .= χ
    initwithrand!(T)
    @tensor T[1,2,3] := T[1,2,3] + T[3,2,1]
    return T
 end

function iterate(iter::rotsymCTMIterable, state::rotsymCTMState)
    @unpack A, χ = iter
    @unpack C, T, oldsvdvals, diffs = state
    #grow
    @tensor begin
        Cp[1,2,3,4]   := C[-1,-2]   * T[1,-3,-1] *
                         T[-2,-4,4] * A[2,3,-4,-3]
        Tp[1,2,3,4,5] := T[1,-1,5]  * A[3,4,-1,2]
    end

    #renormalize
    Z = tensorsvd(Cp, ((1,2),(3,4)), svdtrunc = svdtrunc_maxχ(χ))[1]
    @tensor begin
        Ctmp[1,2] := Cp[-1,-2,-3,-4]  * Z[-1,-2,1] * Z'[-4,-3,2]
        Ttmp[1,2,3] := Tp[-1,-2,2,-3,-4] * Z'[-1,-2,1] * Z[-4,-3,3]
    end

    #symmetrize
    @tensor begin
        C[1,2]   = Ctmp[1,2] + Ctmp[2,1]
        T[1,2,3] = Ttmp[1,2,3] + Ttmp[3,2,1]
    end
    _, s, = tensorsvd(C)
    vals = diag(s)
    maxval = maximum(vals)
    apply!(C, x -> x .= x ./ maxval)
    apply!(T, x -> x .= x ./ maxval)
    normalize!(vals,1)

    #compare
    push!(diffs, sum(abs, oldsvdvals - vals))
    oldsvdvals[:] = vals
    return state, state
end

ctm(A::T, Asz::T, χ::Int; kwargs...) where {T<:AbstractTensor} =  rotsymctm(A,Asz,χ;kwargs...)

function rotsymctm(A::AbstractTensor{S,4}, Asz::AbstractTensor{S,4}, χ::Int;
                                    Cinit::Union{Nothing, AbstractTensor{S,2}} = nothing,
                                    Tinit::Union{Nothing, AbstractTensor{S,3}} = nothing,
                                    tol::Float64 = 1e-13,
                                    maxit::Int = 5000,
                                    period::Int = 100,
                                    verbose::Bool = true,
                                    log::Bool = true) where S
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
    if log
        return (state.C, state.T, (χ = χ, n_it = it, diffs = state.diffs))
    end
    return  (state.C, state.T)

end
