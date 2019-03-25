#=
    Following Corboz, Rice and Troyer in **Competing states in the t-J model: uniform d-wave state versus stripe state**
    p.7
=#

function isommetries(Cs, Ts, As, Ps, Pts, (x,y), χ)
    C1 = Cs[1][x,y]
    C2  = Cs[2][x+3,y]
    C3  = Cs[3][x+3,y+3]
    C4  = Cs[4][x,y+3]
    T11 = Ts[1][x+1,y]
    T12 = Ts[1][x+2,y]
    T21 = Ts[2][x+3,y+1]
    T22 = Ts[2][x+3,y+2]
    T31 = Ts[3][x+2,y+3]
    T32 = Ts[3][x+1,y+3]
    T41 = Ts[4][x,y+2]
    T42 = Ts[4][x,y+1]
    A11  = As[x+1,y+1]
    A21  = As[x+2,y+1]
    A12  = As[x+1,y+2]
    A22  = As[x+2,y+2]

    @tensor begin
        upperhalf[1,2,3,4] := C1[-1,-2] * T42[1,-3,-1] *
                              T11[-2,-4,-5] * A[2,-6,-4,-3] *
                              T12[-5,-7,-8] * A[3,-9,-7,-6] *
                              C2[-8,-10] * T21[-10,-9,4]
        lowerhalf[1,2,3,4] := C3[-1,-2] * T22[4,-3,-1] *
                              T31[-2,-4,-5] * A[-4,-6,3,-3] *
                              T32[-5,-7,-8] * A[-7,-9,2,-6] *
                              C4[-8,-10] * T41[-10,-9,1]
    end

    rup, qup = tensorrq(upperhalf,((1,2),(3,4)))
    rdn, qdn = tensorrq(lowerhalf,((1,2),(3,4)))

    @tensor ruprdn[1,2] :=  rup[-1,-2,1] * rdn[-1,-2,2]
    u, s, v = tensorsvd(ruprdn, svdtrunc = svdtrunc_maxχ(χ))

    sqrts = apply!(s, x-> x.= sqrt.(s))

    @tensor begin
        p[1,2,3]  := rdn[1,2,-1] * v'[-2,-1] * sqrts[-2,2]
        pt[1,2,3] := rup[1,2,-1] * u'[-2,-1] * sqrts[-2,2]
    end
    return p, pt #p[x,y+1], pt[x,y+1]
end
