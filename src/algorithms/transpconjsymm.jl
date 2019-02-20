struct transconjCTMIterable{T,
        TA<:AbstractTensor{T,4},
        TC<:AbstractTensor{T,2},
        TT<:AbstractTensor{T,3}} <: AbstractCTMIterable
    A::TA
    χ::Int
    Cinit::Union{TC,Nothing}
    Tsinit::Union{Tuple{TT,TT},Nothing}
end

function transconjctmiterable(A::DTensor{T,4}, χ::Int,
        Cinit = nothing,
        Tsinit = nothing) where T
    transconjCTMIterable{T,DTensor{T,4},DTensor{T,2},DTensor{T,3}}(A, χ, Cinit, Tsinit)
end

function transconjctmiterable(A::DASTensor{T,4,SYM,CHS,SS,CH}, χ::Int,
        Cinit = nothing,
        Tsinit = nothing) where {T,N,SYM,CHS,SS,CH}
    TC = DASTensor{T,2,SYM,CHS,SS,CH}
    TT = DASTensor{T,3,SYM,CHS,SS,CH}
    TA = DASTensor{T,4,SYM,CHS,SS,CH}
    transconjCTMIterable{T,TA,TC,TT}(A, χ, Cinit, Tsinit)
end

struct transconjCTMState{S,
        TA <: AbstractTensor,
        TC <: AbstractTensor,
        TT <: AbstractTensor} <: AbstractCTMState
    C::TC
    Ts::Tuple{TT,TT}
    oldsvdvals::Vector{S}
    diffs::Vector{S}
    n_it::Ref{Int}
end


function iterate(iter::transconjCTMIterable{S,TA,TC,TT}) where {S,TA,TC,TT}
    @unpack A, χ, Cinit, Tsinit = iter
    C = isnothing(Cinit) ? initializeC(A,χ) : Cinit
    Ts = isnothing(Tsinit) ? (initializeT(A,χ),initializeT(A,χ)) : Tsinit

    l = ifelse(C isa DASTensor, 2χ, χ)
    oldsvdvals = zeros(S,l)
    state = transconjCTMState{S,TA,TC,TT}(C, Ts, oldsvdvals, [], Ref(0))
    return state, state
end

function iterate(iter::transconjCTMIterable, state::transconjCTMState)
    @unpack A, χ = iter
    @unpack C, Ts, oldsvdvals, diffs, n_it = state
    T1, T2 = Ts

    xmove!((C,T2),(T1,A), χ)
    ymove!((C,T1),(T2,A), χ)

    vals = diag(tensorsvd(C)[2])
    maxval = maximum(vals)
    apply!(C, x -> x .= x ./ maxval)
    apply!(T1, x -> x .= x ./ norm(T1))
    apply!(T2, x -> x .= x ./ norm(T2))
    normalize!(vals,1)

    #compare
    push!(diffs, sum(abs, oldsvdvals - vals))
    oldsvdvals[:] = vals
    n_it[] += 1

    return state, state
end

function xmove!((C,T2),(T1,A), χ)
    # grow
    @tensor begin
        Cp[1,2,3]      := C[1,-1]    * T1[-1,2,3]
        T2p[1,2,3,4,5] := T2[1,-1,5] * A[2,3,4,-1]
    end

    #renormalize
    U, = tensorsvd(Cp, ((1,2),(3,)), svdtrunc = svdtrunc_maxχ(χ))
    @tensor begin
        C[1,2]    = U'[-1,-2,1] * Cp[-1,-2,2]
        T2[1,2,3] = U'[-1,-2,1] * T2p[-1,-2,2,-3,-4] * U[-4,-3,3]
    end
end

function ymove!((C,T1),(T2,A), χ)
    # grow
    @tensor begin
        Cp[1,2,3]      := C[-1,3]    * T2[1,2,-1]
        T1p[1,2,3,4,5] := T1[1,-1,5] * A[3,4,-1,2]
    end

    #renormalize
    u,s,Vd = tensorsvd(Cp, ((1,),(2,3)), svdtrunc = svdtrunc_maxχ(χ))
    @tensor begin
        C[1,2]    = Cp[1,-1,-2] * Vd'[2,-1,-2]
        T1[1,2,3] = Vd[1,-1,-2] * T1p[-2,-1,2,-3,-4] * Vd'[3,-3,-4]
    end
end
