#=
    Exploring Corner transfer matrices ...
    p 16
=#

struct fonedirCTMIterable{T,
        TA <: AbstractTensor{T,4},
        TC <: AbstractTensor{T,2},
        TT <: AbstractTensor{T,3}} <: AbstractCTMIterable
    A::TA
    χ::Int
    Csinit::Union{NTuple{2,TC},Nothing}
    Tsinit::Union{NTuple{3,TT},Nothing}
end

function fonedirctmiterable(A::DTensor{T,4}, χ::Int, Csinit = nothing, Tsinit = nothing) where T
    fonedirCTMIterable{T,DTensor{T,4},DTensor{T,2},DTensor{T,3}}(A, χ, Csinit, Tsinit)
end

function fonedirctmiterable(A::DASTensor{T,4,SYM,CHS,SS,CH}, χ::Int,
        Csinit = nothing, Tsinit = nothing) where {T,N,SYM,CHS,SS,CH}
    TC = DASTensor{T,2,SYM,CHS,SS,CH}
    TT = DASTensor{T,3,SYM,CHS,SS,CH}
    TA = DASTensor{T,4,SYM,CHS,SS,CH}
    fonedirCTMIterable{T,TA,TC,TT}(A, χ, Csinit, Tsinit)
end

struct fonedirCTMState{S,
        TA <: AbstractTensor,
        TC <: AbstractTensor,
        TT <: AbstractTensor} <: AbstractCTMState
    Cs::NTuple{2,TC}
    Ts::NTuple{3,TT}
    oldsvdvals::Vector{S}
    diffs::Vector{S}
    n_it::Ref{Int}
end


function iterate(iter::fonedirCTMIterable{S,TA,TC,TT}) where {S,TA,TC,TT}
    @unpack A, χ, Tsinit, Csinit = iter
    Cs = Csinit == nothing ? ntuple(i -> initializeC(A,χ), 2) : deepcopy(Csinit)
    Ts = Tsinit == nothing ? ntuple(i -> initializeT(A,χ), 3) : deepcopy(Tsinit)

    l = ifelse(TA <: DASTensor, 2χ, χ)
    oldsvdvals = zeros(S,l)
    state = fonedirCTMState{S,TA,TC,TT}(Cs, Ts, oldsvdvals, [], Ref(0))
    return state, state
end

function iterate(iter::fonedirCTMIterable, state::fonedirCTMState{S,TA,TC,TT}) where {S,TA,TC,TT}
    @unpack A, χ = iter
    @unpack Cs, Ts, oldsvdvals, diffs, n_it = state
    C1, C2 = Cs
    T1, T2, T4 = Ts
    #=
        [C1] - [T1] - [C2]
          |      |      |
        [T4] - [ A] - [T2]
          |      |      |
        [C1]*- [T1]*- [C2]*
    =#

    #xmove
    fonedirxmove!(Cs, Ts, A, χ)

    #ymove
    fonedirymove!(Cs, Ts, A, χ)

    @tensor C[1,2] := C1[1,-1] * C2[-1,2]
    vals  = diag(tensorsvd(C)[2])
    normalize!(vals,1)

    #compare
    push!(diffs, sum(abs, oldsvdvals - vals))
    oldsvdvals[:] = vals
    n_it[] += 1

    return state, state
end

function fonedirxmove!((C1,C2), (T1,T2,T4), A, χ)
    #=
        [C1] - [T1] - [C2]
          |      |      |
        [T4] - [ A] - [T2]
          |      |      |
        [C1]*- [T1]*- [C2]*
    =#
    @tensor begin
        C1p[1,2,3]     := C1[1,-1]   * T1[-1,2,3]
        T4p[1,2,3,4,5] := T4[1,-1,5] * A[2,3,4,-1]
        C2p[1,2,3]     := T1[1,2,-1]  * C2[-1,3]
        T2p[1,2,3,4,5] := A[4,-1,2,3] * T2[1,-1,5]
    end

    #renormalize
    @tensor begin
        C1pC2p[1,2,3,4] := C1p[1,2,-1] * C2p[-1,3,4]
        C1pC2pC2pC1p[1,2,3,4] := C1pC2p[1,2,-1,-2] * C1pC2p'[3,4,-1,-2]
        C2pC1pC1pC2p[1,2,3,4] := C1pC2p[-1,-2,1,2] * C1pC2p'[-1,-2,3,4]
    end

    U, = tensoreig(C1pC2pC2pC1p, ((1,2),(3,4)), truncfun = svdtrunc_maxχ(χ))
    V, = tensoreig(C1pC2pC2pC1p, ((1,2),(3,4)), truncfun = svdtrunc_maxχ(χ))'

    @tensor begin
        C1[1,2]   = U'[-1,-2,1] * C1p[-1,-2,2]
        T4[1,2,3] = U'[-1,-2,1] * T4p[-1,-2,2,-4,-3] * U[-3,-4,3]
        C2[1,2]   = C2p[1,-1,-2] * V[-2,-1,2]
        T2[1,2,3] = V'[-1,-2,1] * T2p[-1,-2,2,-4,-3] * V[-3,-4,3]
    end

    mval = maximum(diag(tensorsvd(C1)[2]))
    apply!(C1, x -> x .= x ./ mval)
    mval = maximum(diag(tensorsvd(C2)[2]))
    apply!(C2, x -> x .= x ./ mval)
    apply!(T4, x -> x .= x ./ sqrt(norm(T4)))
    apply!(T2, x -> x .= x ./ sqrt(norm(T2)))
end

function fonedirymove!((C1,C2), (T1,T2,T4), A, χ)
    #=
        [C1] - [T1] - [C2]
          |      |      |
        [T4] - [ A] - [T2]
          |      |      |
        [C1]*- [T1]*- [C2]*
    =#
    @tensor begin
        C1p[1,2,3]     := T4[1,2,-1]  * C1[-1,3]
        T1p[1,2,3,4,5] := A[3,4,-1,2] * T1[1,-1,5]
        C2p[1,2,3]     := C2[1,-1]    * T2[-1,2,3]
    end

    #renormalize
    @tensor begin
        C2pC2pC1pC1p[3,4,1,2] := C1p[1,2,-1] * C1p'[-2,-3,-1] * C2p'[-2,-3,-4] * C2p[3,4,-1]
    end

    P, = tensoreig(C1pC1pC2pC2p, ((1,2),(3,4)), truncfun = svdtrunc_maxχ(χ))'
    Pi = pinv(P)

    @tensor begin
        C1[1,2]   = C1p[1,-1,-2] * P[-2,-1,2]
        T1[1,2,3] = Pi[-1,-2,1] * T1p[-1,-2,2,-4,-3] * P[-3,-4,3]
        C2[1,2]   = C2p[-1,-2,2] * Pi[-1,-2,1]
        #FUUUUU
    end

    mval = maximum(diag(tensorsvd(C1)[2]))
    apply!(C1, x -> x .= x ./ mval)
    mval = maximum(diag(tensorsvd(C2)[2]))
    apply!(C2, x -> x .= x ./ mval)
    apply!(T1, x -> x .= x ./ sqrt(norm(T1)))
end
