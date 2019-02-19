struct rotsymCTMIterable{T, TA <: AbstractTensor{T,4}, TC <: AbstractTensor{T,2}, TT <: AbstractTensor{T,3}}
    A::TA
    χ::Int
    Cinit::Union{TC,Nothing}
    Tinit::Union{TT,Nothing}
end

function rotsymctmiterable(A::DTensor{T,4}, χ::Int, Cinit = nothing, Tinit = nothing) where T
    rotsymCTMIterable{T,DTensor{T,4},DTensor{T,2},DTensor{T,3}}(A, χ, Cinit, Tinit)
end

function rotsymctmiterable(A::DASTensor{T,4,SYM,CHS,SS,CH}, χ::Int,
        Cinit = nothing, Tinit = nothing) where {T,N,SYM,CHS,SS,CH}
    TC = DASTensor{T,2,SYM,CHS,SS,CH}
    TT = DASTensor{T,3,SYM,CHS,SS,CH}
    TA = DASTensor{T,4,SYM,CHS,SS,CH}
    rotsymCTMIterable{T,TA,TC,TT}(A, χ, Cinit, Tinit)
end

struct rotsymCTMState{S, TA <: AbstractTensor, TC <: AbstractTensor, TT <: AbstractTensor}
    C::TC
    T::TT
    oldsvdvals::Vector{S}
    diffs::Vector{S}
end


function iterate(iter::rotsymCTMIterable{S,TA,TC,TT}) where {S,TA,TC,TT}
    @unpack A, χ, Cinit, Tinit = iter
    if isnothing(Cinit)
        C = initializeC(A,χ)
        @tensor C[1,2] := C[1,2] + C[2,1]
    else
        C = Cinit
    end
    if isnothing(Tinit)
        T = initializeT(A,χ)
        @tensor T[1,2,3] := T[1,2,3] + T[3,2,1]
    else
        T = Tinit
    end

    l = ifelse(C isa DASTensor, 2χ, χ)
    oldsvdvals = zeros(S,l)
    state = rotsymCTMState{S,TA,TC,TT}(C, T, oldsvdvals, [])
    return state, state
end

function iterate(iter::rotsymCTMIterable, state::rotsymCTMState)
    @unpack A, χ = iter
    @unpack C, T, oldsvdvals, diffs = state
    #grow
    @tensor begin
        Cp[1,2,3,4]   := C[-1,-2]   * T[1,-3,-1] *
                         T[-2,-4,4] * A[2,3,-4,-3]
        Tp[1,2,3,4,5] := T[1,-1,5]  * A[3,4,-1 ,2]
    end

    #renormalize
    Z = tensorsvd(Cp, ((1,2),(3,4)), svdtrunc = svdtrunc_maxχ(χ))[1]
    @tensor begin
        Ctmp[1,2]   := Cp[-1,-2,-3,-4]   * Z[-1,-2,1] * Z'[-4,-3,2]
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
