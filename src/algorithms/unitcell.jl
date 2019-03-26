#=
    Following Corboz, Rice and Troyer in **Competing states in the t-J model: uniform d-wave state versus stripe state**
    p.7
=#

struct UnitCell{TA,N}
    tensors::Matrix{TA}
    lvecs::NTuple{2,SVector{2,Int}}
    m::SArray{Tuple{2,2},Float64,2,4}
    UnitCell(ts, lvecs) = new{eltype(ts)}(ts,lvecs, inv(hcat(lvecs...)))
end


function shiftcoordinates(v::SVector{2,Int}, m, (l1,l2))
    s1, s2 = round.(Int,m * v, RoundDown)
    x, y = v - s1*l1 - s2*l2 + one.(v)
    return x,y
end

function Base.getindex(uc::UnitCell,x::Int,y::Int)
    @unpack m, lvecs, tensors = uc
    x, y = shiftcoordinates(SVector{2,Int}(x,y), m, lvecs)
    return tensors[x,y]
end

function Base.setindex!(uc::UnitCell, A, x::Int,y::Int)
    @unpack m, lvecs, tensors = uc
    x, y = shiftcoordinates(SVector{2,Int}(x,y), m, lvecs)
    tensors[x,y] = A
    return uc
end

struct UnitCellIterator{N,TA}
    A ::UnitCell{TA,N}
    χ::Int
    uniques::NTuple{N,Tuple{Int,Int}}
end

struct UnitCellState{N,TA,TC,TT,TP}
    C1::UnitCell{TC}
    C2::UnitCell{TC}
    C3::UnitCell{TC}
    C4::UnitCell{TC}
    T1::UnitCell{TT}
    T2::UnitCell{TT}
    T3::UnitCell{TT}
    T4::UnitCell{TT}
    P ::UnitCell{TP}
    PT::UnitCell{TP}

    oldsvdvals::Vector{Float64}
    diffs::Vector{Float64}
    n_it::Ref{Int}
end

function iterate(iter::UnitCellIterator{N,TA}) where {N,TA}
    @unpack A, χ, uniques = iter
    Ctmp = initializeC(A,χ)
    Ttmp = initializeT(A,χ)
    Ptmp = checked_similar_from_indices(nothing,
            eltype(A[1,1]), (1,2), (1,), (1,3,2), A[1,1], Ctmp, :Y, :Y)

    TC = typeof(Ctmp)
    TT = typeof(Ttmp)
    TP = typeof(Ptmp)
    st = size(A.tensors)
    C1 = Matrix{TC}(undef, st...)
    C2 = Matrix{TC}(undef, st...)
    C3 = Matrix{TC}(undef, st...)
    C4 = Matrix{TC}(undef, st...)
    T1 = Matrix{TT}(undef, st...)
    T2 = Matrix{TT}(undef, st...)
    T3 = Matrix{TT}(undef, st...)
    T4 = Matrix{TT}(undef, st...)
    PT = Matrix{TP}(undef, st...)
    P  = Matrix{TP}(undef, st...)
    for (x,y) in uniques
        C1[x,y] = initwithrand!(similar(Ctmp))
        C2[x,y] = initwithrand!(similar(Ctmp))
        C3[x,y] = initwithrand!(similar(Ctmp))
        C4[x,y] = initwithrand!(similar(Ctmp))
        T1[x,y] = initwithrand!(similar(Ttmp))
        T2[x,y] = initwithrand!(similar(Ttmp))
        T3[x,y] = initwithrand!(similar(Ttmp))
        T4[x,y] = initwithrand!(similar(Ttmp))
        PT[x,y] = initwithrand!(similar(Ptmp))
        P[x,y]  = initwithrand!(similar(Ptmp))
    end

    l = ifelse(C isa DASTensor, 2*4χ, χ)
    oldsvdvals = zeros(real(S),l)
    state = rotsymCTMState{real(S),TA,TC,TT}(C, T, oldsvdvals, [], Ref(0))
    return state, state
end

function iterate(ucit::UnitCellIterator, ucstate::UnitCellState)
    @unpack A, χ, uniques = iter
    @unpack oldsvdvals, diffs, n_it, C1, C2, C3, C4, T1, T2, T3, T4 = state

    leftmove!(ucit, ucstate)
    upmove!(ucit, ucstate)
    rightmove!(ucit, ucstate)
    downmove!(ucit, ucstate)

    vals = copy(oldsvdvals)
    for (i,(x,y)) in enumerate(uniques)
        @tensor ρ[1,2] := C1[x-1,y-1][ 1,-1] * C2[x+1,y-1][-1,-2] *
                          C3[x+1,y+1][-2,-3] * C4[x-1,y+1][-3, 2]
        _, s, = tensorsvd(ρ)
        val = diag(s)
        apply!.((C1[x-1,y-1], C2[x+1,y-1], C3[x+1,y+1], C4[x-1,y+1],
                 T1[x,y-1], T2[x+2,y-1], T3[x+1, y+2], T4[x-1,y]),
                x -> x .= x./ maxval)
        vals[1:χ .+ (i-1)] .= diag(s)
    end
    normalize!(vals,1)
    #compare
    push!(diffs, sum(abs, oldsvdvals - vals))
    oldsvdvals[:] = vals
    n_it[] += 1

    return state, state
end


"""
    numering left to right
"""
function horizontalcut(ucit, ucstate,(x,y))
    @unpack A = ucit
    @unpack C1, C2, C3, C4, T1, T2, T3, T4 = ucstate
    @tensor begin
        upperhalf[1,2,3,4] := C1[x  ,y][-1,-2]   * T4[x  ,y+1][1,-3,-1] *
                              T1[x+1,y][-2,-4,-5] * A[x+1,y+1][2,-6,-4,-3] *
                              T1[x+2,y][-5,-7,-8] * A[x+2,y+1][3,-9,-7,-6] *
                              C2[x+3,y][-8,-10]  * T2[x+3,y+1][-10,-9,4]

        lowerhalf[1,2,3,4] := C3[x+3,y+3][-1,-2]   * T2[x+3,y+2][4,-3,-1] *
                              T3[x+2,y+3][-2,-4,-5] * A[x+2,y+2][-4,-3,3,-6] *
                              T3[x+1,y+3][-5,-7,-8] * A[x+1,y+2][-7,-6,2,-9] *
                              C4[x  ,y+3][-8,-10]  * T4[x  ,y+2][-10,-9,1]
    end
    return upperhalf, lowerhalf
end

"""
    numbering bottom to top
"""
function verticalcut(ucit, ucstate,(x,y))
    @unpack A = ucit
    @unpack C1, C2, C3, C4, T1, T2, T3, T4 = ucstate
    @tensor begin
        lefthalf[1,2,3,4] :=  C4[x  ,y][-1,-2]   * T3[x  ,y+1][1,-3,-1] *
                              T4[x+1,y][-2,-4,-5] * A[x+1,y+1][-3,2,-6,-4] *
                              T4[x+2,y][-5,-7,-8] * A[x+2,y+1][-6,3,-9,-7] *
                              C1[x+3,y][-8,-10]  * T1[x+3,y+1][-10,-9,4]
        righthalf[1,2,3,4] := C2[x+3,y+3][-1,-2]   * T1[x+3,y+2][4,-3,-1] *
                              T2[x+2,y+3][-2,-4,-5] * A[x+2,y+2][-6,-4,-3,3] *
                              T2[x+1,y+3][-5,-7,-8] * A[x+1,y+2][-9,-7,-6,2] *
                              C3[x  ,y+3][-8,-10]  * T3[x  ,y+2][-10,-9,1]
    end
    return lefthalf, righthalf
end

"""
"""
function leftisommetries!(ucit, ucstate, (x,y), χ)
    @unpack χ = ucit
    @unpack P, PT = ucstate
    upperhalf, lowerhalf = horizontalcut(ucit, ucstate)

    rup, qup = tensorrq(upperhalf,((1,2),(3,4)))
    rdn, qdn = tensorrq(lowerhalf,((1,2),(3,4)))

    @tensor ruprdn[1,2] :=  rup[-1,-2,1] * rdn[-1,-2,2]
    u, s, vd = tensorsvd(ruprdn, svdtrunc = svdtrunc_maxχ(χ))

    sqrts = apply!(s, x-> x.= sqrt.(s))

    @tensor begin
        P[x, y+1][1,2,3] = rdn[1,2,-1] * vd'[-2,-1] * sqrts[-2,3]
        Pt[x,y+1][1,2,3] = rup[1,2,-1] * u'[-2, -1] * sqrts[-2,3]
    end
end

"""
"""
function rightisommetries!(ucit, ucstate, (x,y), χ)
    @unpack χ = ucit
    @unpack P, PT = ucstate
    upperhalf, lowerhalf = horizontalcut(ucit, ucstate)

    qup, rup = tensorqr(upperhalf,((1,2),(3,4)))
    qdn, rdn = tensorqr(lowerhalf,((1,2),(3,4)))

    @tensor ruprdn[1,2] :=  rup[1,-1,-2] * rdn[2,-1,-2]
    u, s, v = tensorsvd(ruprdn, svdtrunc = svdtrunc_maxχ(χ))

    sqrts = apply!(s, x-> x.= sqrt.(s))

    @tensor begin
        P[ x+4,y+1][1,2,3] = rdn[-1,1,2] * v'[-2,-1] * sqrts[-2,3]
        Pt[x+4,y+1][1,2,3] = rup[-1,1,2] * u'[-2,-1] * sqrts[-2,3]
    end
end

"""
"""
function downisommetries!(ucit, ucstate, (x,y), χ)
    @unpack χ = ucit
    @unpack P, PT = ucstate
    lefthalf, righthalf = verticalcut(ucit, ucstate)

    rl, qup = tensorrq(lefthalf,((1,2),(3,4)))
    rr, qdn = tensorrq(righthalf,((1,2),(3,4)))

    @tensor rlrr[1,2] :=  rl[-1,-2,1] * rr[-1,-2,2]
    u, s, v = tensorsvd(rlrr, svdtrunc = svdtrunc_maxχ(χ))
    sqrts = apply!(s, x-> x.= sqrt.(s))

    @tensor begin
        P[ x+1,y+4][1,2,3] = rr[1,2,-1] * v'[-2,-1] * sqrts[-2,3]
        Pt[x+1,y+4][1,2,3] = rl[1,2,-1] * u'[-2,-1] * sqrts[-2,3]
    end
end

"""
"""
function upisommetries!(ucit, ucstate, (x,y), χ)
    @unpack χ = ucit
    @unpack P, PT = ucstate
    lefthalf, righthalf = verticalcut(ucit, ucstate)

    ql, rl = tensorqr(lefthalf,((1,2),(3,4)))
    qr, rr = tensorqr(righthalf,((1,2),(3,4)))

    @tensor rlrr[1,2] :=  rl[1,-1,-2] * rr[2,-1,-2]
    u, s, v = tensorsvd(rlrr, svdtrunc = svdtrunc_maxχ(χ))
    sqrts = apply!(s, x-> x.= sqrt.(s))

    @tensor begin
        P[ x+1,y][1,2,3] = rl[-1,1,2] * v'[-2,-1] * sqrts[-2,3]
        Pt[x+1,y][1,2,3] = rr[-1,1,2] * u'[-2,-1] * sqrts[-2,3]
    end
end

function leftmove!(ucit::UnitCellIterator, ucstate)
    @unpack uniques, χ = ucit
    for (x,y) in uniques
        leftisommetries!(ucit, ucstate, (x,y), χ)
    end
    for (x,y) in unique
        leftmove!(ucit, ucstate, (x,y))
    end
end

function rightmove!(ucit::UnitCellIterator, ucstate)
    @unpack uniques, χ = ucit
    for (x,y) in uniques
        rightisommetries!(ucit, ucstate, (x,y), χ)
    end
    for (x,y) in unique
        rightmove!(ucit, ucstate, (x,y))
    end
end

function upmove!(ucit::UnitCellIterator, ucstate)
    @unpack uniques, χ = ucit
    for (x,y) in uniques
        upisommetries!(ucit, ucstate, (x,y), χ)
    end
    for (x,y) in unique
        upmove!(ucit, ucstate, (x,y))
    end
end

function downmove!(ucit::UnitCellIterator, ucstate)
    @unpack uniques, χ = ucit
    for (x,y) in uniques
        downisommetries!(ucit, ucstate, (x,y), χ)
    end
    for (x,y) in unique
        downmove!(ucit, ucstate, (x,y))
    end
end

function leftmove!(ucit::UnitCellIterator, ucstate, (x,y))
    @unpack C1, T4, C4, Pt, P = ucstate
    @unpack A = ucit

    @tensor begin
        C1[x+1,y][1,2]   = C1[x,y][-1,-2] * T1[x+1,y][-2,-3,2] * Pt[x,y][-1,-3,1]
        T4[x+1,y][1,2,3] = T4[x,y][-1,-2,-3] * A[x+1,y][-4,2,-5,-2] *
                           P[x,y-1][-3,-5,3] * Pt[x,y][-1,-4,1]
        C4[x+1,y][1,2]   = C4[x,y][-1,-2] * T3[x+1,y][1,-3,-1] * P[x,y-1][-2,-3,2]
    end
end

function rightmove!(ucit::UnitCellIterator, ucstate, (x,y))
    @unpack C2, T2, C3, Pt, P = ucstate
    @unpack A = ucit

    @tensor begin
        C2[x-1,y][1,2]   = C2[x,y][-1,-2] * T1[x-1,y][1,-3,-1] * Pt[x,y][-3,-2,2]
        T2[x-1,y][1,2,3] = T2[x,y][-1,-2,-3] * A[x-1,y][-5,-2,-4,2] *
                           P[x,y-1][-4,-1,1] * Pt[x,y][-5,-3,3]
        C3[x-1,y][1,2]   = C3[x,y][-1,-2] * T3[x-1,y][-2,-3,2] * P[x,y-1][-3,-1,1]
    end
end

function upmove!(ucit::UnitCellIterator, ucstate, (x,y))
    @unpack C2, T2, C3, Pt, P = ucstate
    @unpack A = ucit

    @tensor begin
        C1[x,y+1][1,2]   = C1[x,y][-1,-2] * T4[x,y][1,-3,-1] * Pt[x,y][-3,-2,2]
        T1[x,y+1][1,2,3] = T1[x,y][-1,-2,-3] * A[x,y][2,-5,-2,-4] *
                           P[x,y-1][-4,-1,1] * Pt[x,y][-4,-3,3]
        C2[x,y+1][1,2]   = C2[x,y][-1,-2] * T2[x,y][-2,-3,2] * P[x,y-1][-3,-1,1]
    end
end

function downmove!(ucit::UnitCellIterator, ucstate, (x,y))
    @unpack C2, T2, C3, Pt, P = ucstate
    @unpack A = ucit

    @tensor begin
        C3[x-1,y][1,2]   = C3[x,y][-1,-2] * T2[x,y-1][1,-3,-1] * Pt[x,y][-3,-1,1]
        T3[x-1,y][1,2,3] = T3[x,y][-1,-2,-3] * A[x-1,y][-2,-4,2,-5] *
                           P[x,y-1][-3,-5,3] * Pt[x,y][-1,-4,1]
        C4[x-1,y][1,2]   = C4[x,y][-1,-2] * T4[x-1,y][-2,-3,2] * P[x,y-1][-1,-3,1]
    end
end
