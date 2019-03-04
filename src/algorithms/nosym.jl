#=
    Following the description in
        _Simulation of two-dimensional quantum systems on an infinite lattice revisited:
        Corner Transfer matrix for tensor contraction_
    by Orus and Vidal

    Implemented here is the directional variant of the CTMRG as outlined
    on page 2 and in Fig2.
    DOI: 10.1103/PhysRevB.80.094403
=#

struct CTMIterable{T,
        TA <: AbstractTensor{T,4},
        TC <: AbstractTensor{T,2},
        TT <: AbstractTensor{T,3}} <: AbstractCTMIterable
    A::TA
    χ::Int
    Csinit::Union{NTuple{4,TC},Nothing}
    Tsinit::Union{NTuple{4,TT},Nothing}
end

function ctmiterable(A::DTensor{T,4}, χ::Int, Csinit = nothing, Tsinit = nothing) where T
    CTMIterable{T,DTensor{T,4},DTensor{T,2},DTensor{T,3}}(A, χ, Csinit, Tsinit)
end

function ctmiterable(A::DASTensor{T,4,SYM,CHS,SS,CH}, χ::Int,
        Csinit = nothing, Tsinit = nothing) where {T,N,SYM,CHS,SS,CH}
    TC = DASTensor{T,2,SYM,CHS,SS,CH}
    TT = DASTensor{T,3,SYM,CHS,SS,CH}
    TA = DASTensor{T,4,SYM,CHS,SS,CH}
    CTMIterable{T,TA,TC,TT}(A, χ, Csinit, Tsinit)
end

struct CTMState{S,
        TA <: AbstractTensor,
        TC <: AbstractTensor,
        TT <: AbstractTensor} <: AbstractCTMState
    Cs::NTuple{4,TC}
    Ts::NTuple{4,TT}
    oldsvdvals::Vector{S}
    diffs::Vector{S}
    n_it::Ref{Int}
end


function iterate(iter::CTMIterable{S,TA,TC,TT}) where {S,TA,TC,TT}
    @unpack A, χ, Tsinit, Csinit = iter
    Cs = Csinit == nothing ? ntuple(i -> initializeC(A,χ), 4) : deepcopy(Csinit)
    Ts = Tsinit == nothing ? ntuple(i -> initializeT(A,χ), 4) : deepcopy(Tsinit)

    l = ifelse(TA <: DASTensor, 2χ, χ)
    oldsvdvals = zeros(real(S),l)
    state = CTMState{real(S),TA,TC,TT}(Cs, Ts, oldsvdvals, [], Ref(0))
    return state, state
end

function iterate(iter::CTMIterable, state::CTMState{S,TA,TC,TT}) where {S,TA,TC,TT}
    @unpack A, χ = iter
    @unpack Cs, Ts, oldsvdvals, diffs, n_it = state
    C1, C2, C3, C4 = Cs
    T1, T2, T3, T4 = Ts
    #=
        [C1] - [T1] - [C2]
          |      |      |
        [T4] - [ A] - [T2]
          |      |      |
        [C4] - [T3] - [C3]
    =#

    #leftmove
    leftmove!((C1,T4,C4),(T1,A,T3),χ)

    # upmove
    upmove!((C1,T1,C2),(T4,A,T2),χ)

    #rightmove
    rightmove!((C3,T2,C2),(T3,A,T1),χ)

    #downmove
    downmove!((C4,T3,C3),(T4,A,T2),χ)

    @tensor C[1,2] := C1[1,-1] * C2[-1,-2] * C3[-2,-3] * C4[-3,2]
    vals  = diag(tensorsvd(C)[2])
    normalize!(vals,1)

    #compare
    push!(diffs, sum(abs, oldsvdvals - vals))
    oldsvdvals[:] = vals
    n_it[] += 1

    return state, state
end

function leftmove!((C1,T4,C4), (T1,A,T3), χ)
    #=
    leftmove:
        [C1]--[T1]--
          |    |
        [T4]--[ A]--
          |    |
        [C4]--[T3]--
    =#

    @tensor begin
        C1p[1,2,3]     := C1[1,-1]   * T1[-1,2,3]
        T4p[1,2,3,4,5] := T4[1,-1,5] * A[2,3,4,-1]
        C4p[1,2,3]     := C4[-1,3]   * T3[1,2,-1]
    end

    #renormalize
    @tensor begin
        C1pC1p[1,2,3,4] := C1p[1,2,-1] * C1p'[4,3,-1]
        C4pC4p[1,2,3,4] := C4p[-1,2,1] * C4p'[-1,3,4]
        CCpCC[1,2,4,3]  := C1pC1p[1,2,3,4] + C4pC4p[1,2,3,4]
    end

    CCpCCmat, reshaper = fuselegs(CCpCC,((1,2),(3,4)))
    E, = tensoreig(CCpCCmat,truncfun = svdtrunc_maxχ(χ))
    U = splitlegs(E, ((1,1,1),(1,1,2),2), reshaper...)

    @tensor begin
        C1[1,2]   = U'[-1,-2,1] * C1p[-1,-2,2]
        T4[1,2,3] = U'[-1,-2,1] * T4p[-1,-2,2,-4,-3] * U[-3,-4,3]
        C4[1,2]   = C4p[1,-1,-2] * U[-2,-1,2]
    end

    mval = maximum(diag(tensorsvd(C1)[2]))
    apply!(C1, x -> x .= x ./ mval)
    mval = maximum(diag(tensorsvd(C4)[2]))
    apply!(C4, x -> x .= x ./ mval)
    apply!(T4, x -> x .= x ./ sqrt(norm(T4)))
end

function rightmove!((C3,T2,C2),(T3,A,T1),χ)
    #=
    rightmove:
      --[T1]---[C2]
         |      |
      --[ A]---[T2]
         |      |
      --[T3]---[C3]
    =#

    @tensor begin
        C2p[1,2,3]     := T1[1,2,-1]  * C2[-1,3]
        T2p[1,2,3,4,5] := A[4,-1,2,3] * T2[1,-1,5]
        C3p[1,2,3]     := C3[1,-1]    * T3[-1,2,3]
    end

    #renormalize
    @tensor begin
        C2pC2p[1,2,3,4] := C2p[-1,2,1] * C2p'[-1,3,4]
        C3pC3p[1,2,3,4] := C3p[1,2,-1] * C3p'[4,3,-1]
        CCpCC[1,2,4,3]  := C2pC2p[1,2,3,4] + C3pC3p[1,2,3,4]
    end

    CCpCCmat, reshaper = fuselegs(CCpCC,((1,2),(3,4)))
    E, = tensoreig(CCpCCmat,truncfun = svdtrunc_maxχ(χ))
    U = splitlegs(E, ((1,1,1),(1,1,2),2), reshaper...)

    @tensor begin
        C2[1,2]   = C2p[1,-1,-2] * U'[-2,-1,2]
        T2[1,2,3] = U[-1,-2,1] * T2p[-1,-2,2,-4,-3] * U'[-3,-4,3]
        C3[1,2]   = C3p[-1,-2,2] * U[-1,-2,1]
    end

    mval = maximum(diag(tensorsvd(C2)[2]))
    apply!(C2, x -> x .= x ./ mval)
    mval = maximum(diag(tensorsvd(C3)[2]))
    apply!(C3, x -> x .= x ./ mval)
    apply!(T2, x -> x .= x ./ sqrt(norm(T2)))
end

function upmove!((C1,T1,C2),(T4,A,T2),χ)
    #=
    upmove:
        [C1]--[T1]--[C2]
          |     |     |
        [T4]--[ A]--[T2]
          |     |     |
    =#

    @tensor begin
        C1p[1,2,3]     := T4[1,2,-1]  * C1[-1,3]
        T1p[1,2,3,4,5] := A[3,4,-1,2] * T1[1,-1,5]
        C2p[1,2,3]     := C2[1,-1]    * T2[-1,2,3]
    end

    #renormalize
    @tensor begin
        C1pC1p[1,2,3,4] := C1p[-1,2,1] * C1p'[-1,3,4]
        C2pC2p[1,2,3,4] := C2p[1,2,-1] * C2p'[4,3,-1]
        CCpCC[1,2,4,3]  := C1pC1p[1,2,3,4] + C2pC2p[1,2,3,4]
    end

    CCpCCmat, reshaper = fuselegs(CCpCC,((1,2),(3,4)))
    E, = tensoreig(CCpCCmat,truncfun = svdtrunc_maxχ(χ))
    U = splitlegs(E, ((1,1,1),(1,1,2),2), reshaper...)

    @tensor begin
        C1[1,2]   = C1p[1,-1,-2] * U'[-2,-1,2]
        T1[1,2,3] = U'[-1,-2,1] * T1p[-1,-2,2,-4,-3] * U[-3,-4,3]
        C2[1,2]   = C2p[-1,-2,2] * U[-1,-2,1]
    end

    mval = maximum(diag(tensorsvd(C1)[2]))
    apply!(C1, x -> x .= x ./ mval)
    mval = maximum(diag(tensorsvd(C2)[2]))
    apply!(C2, x -> x .= x ./ mval)
    apply!(T1, x -> x .= x ./ sqrt(norm(T1)))
end

function downmove!((C4,T3,C3),(T4,A,T2),χ)
    #=
    downmove:
          |     |     |
        [T4]--[ A]--[T2]
          |     |     |
        [C4]--[T3]--[C3]
    =#

    @tensor begin
        C4p[1,2,3]     := C4[1,-1]  * T4[-1,2,3]
        T3p[1,2,3,4,5] := T3[1,-1,5] * A[-1,2,3,4]
        C3p[1,2,3]     := T2[1,2,-1] * C3[-1,3]
    end

    #renormalize
    @tensor begin
        C3pC3p[1,2,3,4] := C3p[-1,2,1] * C3p'[-1,3,4]
        C4pC4p[1,2,3,4] := C4p[1,2,-1] * C4p'[4,3,-1]
        CCpCC[1,2,4,3]  := C3pC3p[1,2,3,4] + C4pC4p[1,2,3,4]
    end

    CCpCCmat, reshaper = fuselegs(CCpCC,((1,2),(3,4)))
    E, = tensoreig(CCpCCmat,truncfun = svdtrunc_maxχ(χ))
    U = splitlegs(E, ((1,1,1),(1,1,2),2), reshaper...)

    @tensor begin
        C3[1,2]   = C3p[1,-1,-2] * U[-2,-1,2]
        T3[1,2,3] = U'[-1,-2,1] * T3p[-1,-2,2,-3,-4] * U[-4,-3,3]
        C4[1,2]   = C4p[-1,-2,2] * U'[-1,-2,1]
    end

    mval = maximum(diag(tensorsvd(C4)[2]))
    apply!(C4, x -> x .= x ./ mval)
    mval = maximum(diag(tensorsvd(C3)[2]))
    apply!(C3, x -> x .= x ./ mval)
    apply!(T3, x -> x .= x ./ sqrt(norm(T3)))
end
