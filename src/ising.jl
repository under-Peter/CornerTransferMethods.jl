const βc = log(1+sqrt(2))/2

function magofβ(β)
    if β > βc
        (1-sinh(2*β)^-4)^(1/8)
    else
        0
    end
end

#Crone Thesis tensors:
function atens(β)
    a = zeros(Float64,2,2,2,2)
    a[1,1,1,1] = a[2,2,2,2] = 1
    cβ = sqrt(cosh(β))
    sβ = sqrt(sinh(β))
    Q = 1/sqrt(2) * [cβ+sβ cβ-sβ; cβ-sβ cβ+sβ]
    @tensor a[1,2,3,4] := a[-1,-2,-3,-4] * Q[-1,1] * Q[-2,2] * Q[-3,3] * Q[-4,4]
    return DTensor(a)
end

function asztens(β)
    a = zeros(Float64,2,2,2,2)
    a[1,1,1,1] = 1
    a[2,2,2,2] = -1
    cβ = sqrt(cosh(β))
    sβ = sqrt(sinh(β))
    Q = 1/sqrt(2) * [cβ+sβ cβ-sβ; cβ-sβ cβ+sβ]
    @tensor a[1,2,3,4] := a[-1,-2,-3,-4] * Q[-1,1] * Q[-2,2] * Q[-3,3] * Q[-4,4]
    return DTensor(a)
end

function isingenvironment(β, χ; fixed::Bool = true)
    C = zeros(Float64,χ,χ)
    T = zeros(Float64,χ,2,χ)
    C[1,1]  = 1
    T[1,1,1] = 1
    if !fixed
        C[2,2] = 1
        T[2,2,2] = 1
    end
    cβ = sqrt(cosh(β))
    sβ = sqrt(sinh(β))
    q = 1/sqrt(2) * [cβ+sβ cβ-sβ; cβ-sβ cβ+sβ]
    Q = zeros(Float64, χ, χ)
    Q[1:2,1:2] = q
    @tensor T[1,2,3] := T[-1,-2,-3] * Q[1,-1] * q[2,-2] * Q[3,-3]
    @tensor C[1,2] := C[-1,-2] * Q[1,-1] * Q[2,-2]
    return (DTensor(C), DTensor(T))
end

function isingenvironmentz2(β, χ; fixed::Bool = true)
    C = ZNTensor{Float64,2,2}(([χ,χ],[χ,χ]),(-1,-1))
    T = ZNTensor{Float64,3,2}(([χ,χ],[1,1],[χ,χ]),(1,-1,1))
    C[(0,0)] = zeros(Float64,χ,χ)
    T[(0,0,0)] = zeros(Float64,χ,1,χ)
    C[(0,0)][1,1] = 1
    T[(0,0,0)][1,1,1] = 1
    if !fixed
        C[(1,1)] = zeros(Float64,χ,χ)
        T[(1,1,1)] = zeros(Float64,χ,1,χ)
        C[(1,1)][1,1] = 1
        T[(1,1,1)][1,1,1] = 1
    end
    q = ZNTensor{Float64,2,2}((0:1,0:1),([1,1],[1,1]),(1,-1))
    qχ = ZNTensor{Float64,2,2}((0:1,0:1),([χ,χ],[χ,χ]),(1,-1))
    cβ = sqrt(cosh(β))
    sβ = sqrt(sinh(β))
    t00 = 2^(-1/2) * (cβ + sβ)
    t01 = 2^(-1/2) * (cβ - sβ)
    q[(0,0)] = reshape([t00],1,1)
    q[(1,1)] = reshape([t00],1,1)
    q[(0,1)] = reshape([t01],1,1)
    q[(1,0)] = reshape([t01],1,1)
    qχ[(0,0)] = zeros(Float64,χ,χ)
    qχ[(1,1)] = zeros(Float64,χ,χ)
    qχ[(0,1)] = zeros(Float64,χ,χ)
    qχ[(1,0)] = zeros(Float64,χ,χ)
    qχ[(0,0)][1,1] = t00
    qχ[(1,1)][1,1] = t00
    qχ[(0,1)][1,1] = t01
    qχ[(1,0)][1,1] = t01
    @tensor T[1,2,3] := T[-1,-2,-3] * qχ[1,-1] * q[2,-2] * qχ[3,-3]
    @tensor C[1,2] := C[-1,-2] * qχ[1,-1] * qχ[2,-2]
    return (C, T)
end

function atensz2(β)
    a = ZNTensor{Float64,4,2}((0:1,0:1,0:1,0:1),([1,1],[1,1],[1,1],[1,1]), (1,1,1,1))
    b = ZNTensor{Float64,2,2}((0:1,0:1),([1,1],[1,1]),(1,-1))
    cβ = reshape([sqrt(cosh(β))],1,1)
    sβ = reshape([sqrt(sinh(β))],1,1)
    a[(0,0,0,0)] = reshape([1],1,1,1,1)
    a[(1,1,1,1)] = reshape([1],1,1,1,1)
    b[(0,0)] = 2^(-1/2) * (cβ + sβ)
    b[(1,1)] = 2^(-1/2) * (cβ + sβ)
    b[(0,1)] = 2^(-1/2) * (cβ - sβ)
    b[(1,0)] = 2^(-1/2) * (cβ - sβ)
    @tensor a[o1,o2,o3,o4] := a[c1,c2,c3,c4] * b[o1,c1] * b[o2,c2] * b[o3,c3] * b[o4,c4]
    return a
end

function asztensz2(β)
    a = ZNTensor{Float64,4,2}((0:1,0:1,0:1,0:1),([1,1],[1,1],[1,1],[1,1]), (1,1,1,1))
    b = ZNTensor{Float64,2,2}((0:1,0:1),([1,1],[1,1]),(1,-1))
    cβ = reshape([sqrt(cosh(β))],1,1)
    sβ = reshape([sqrt(sinh(β))],1,1)
    a[(0,0,0,0)] = reshape([1],1,1,1,1)
    a[(1,1,1,1)] = reshape([-1],1,1,1,1)
    b[(0,0)] = 2^(-1/2) * (cβ + sβ)
    b[(1,1)] = 2^(-1/2) * (cβ + sβ)
    b[(0,1)] = 2^(-1/2) * (cβ - sβ)
    b[(1,0)] = 2^(-1/2) * (cβ - sβ)
    @tensor a[o1,o2,o3,o4] := a[c1,c2,c3,c4] * b[o1,c1] * b[o2,c2] * b[o3,c3] * b[o4,c4]
    return a
end

atenses(β) = (atens(β), asztens(β))
atensesz2(β) = (atensz2(β), asztensz2(β))

# isingpart(β) = partitionfun(ising, β)

magnetisation(state::rotsymCTMState, a, asz) = magnetisation(state.C, state.T, a, asz)
magnetisation((C,T,a,asz)) = magnetisation(C,T,a,asz)

function magnetisation(C::AbstractTensor{S,2}, T::AbstractTensor{S,3}, a::AbstractTensor{S,4}, asz::AbstractTensor{S,4}) where S
    @tensor ctc[1,2,3] := C[1,-1] * T[-1,2,-2] * C[-2,3]
    @tensor begin
        mag[]  := ctc[-1,-2,-3] * T[-3,-4,-5] * ctc[-7,-6,-5] * T[-7,-8,-1] * asz[-2,-4,-6,-8]
        norm[] := ctc[-1,-2,-3] * T[-3,-4,-5] * ctc[-5,-6,-7] * T[-7,-8,-1] * a[-2,-4,-6,-8]
        # norm[] := C[-1,-2] * C[-2,-3] * C[-3,-4] * C[-4,-1]
    end
    return scalar(mag)/scalar(norm)
end

function isingctm(β, χ, fixed::Bool = true; kwargs...)
    cinit, tinit = isingenvironment(β, χ, fixed = fixed)
    a, asz = atenses(β)
    (ctm(a, asz, χ; kwargs...)[2]..., a, asz)
end

function isingctmz2(β, χ, fixed::Bool = true; kwargs...)
    cinit, tinit = isingenvironmentz2(β, χ, fixed = fixed)
    a, asz = atensesz2(β)
    (ctm(a, asz, χ; kwargs...)[2]..., a, asz)
end
