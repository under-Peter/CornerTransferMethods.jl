const βc = log(1+sqrt(2))/2

function magofβ(β)
    if β > βc
        (1-sinh(2*β)^-4)^(1/8)
    else
        0
    end
end

#Crone Thesis tensors:
function isingtensor(β)
    a = zeros(Float64,2,2,2,2)
    a[1,1,1,1] = a[2,2,2,2] = 1
    cβ = sqrt(cosh(β))
    sβ = sqrt(sinh(β))
    Q = 1/sqrt(2) * [cβ+sβ cβ-sβ; cβ-sβ cβ+sβ]
    @tensor a[1,2,3,4] := a[-1,-2,-3,-4] * Q[-1,1] * Q[-2,2] * Q[-3,3] * Q[-4,4]
    return DTensor(a)
end

function isingmagtensor(β)
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

function isingenvironmentz2(β::Float64, χ::Int)
    C = ZNTensor{Float64,2,2}(([χ,χ],[χ,χ]),(-1,-1))
    T = ZNTensor{Float64,3,2}(([χ,χ],[1,1],[χ,χ]),(1,-1,1))
    for i in Iterators.filter(iseven ∘ sum, Iterators.product(charges(C)...))
        C[i] = zeros(Float64,χ,χ)
        C[i][1] = 1
    end
    for i in Iterators.filter(iseven ∘ sum, Iterators.product(charges(T)...))
        T[i] = zeros(Float64,χ,1,χ)
        T[i][1] = 1
    end
    q = ZNTensor{Float64,2,2}((0:1,0:1),([1,1],[1,1]),(1,-1))
    qχ = ZNTensor{Float64,2,2}((0:1,0:1),([χ,χ],[χ,χ]),(1,-1))
    cβ = [sqrt(cosh(β))]
    sβ = [sqrt(sinh(β))]
    q[(0,0)] = reshape(sqrt(2) * cβ, 1,1)
    q[(1,1)] = reshape(sqrt(2) * cβ, 1,1)
    qχ[(0,0)] = zeros(Float64,χ,χ)
    qχ[(1,1)] = zeros(Float64,χ,χ)
    qχ[(0,0)][1] = sqrt(2) * cβ[1]
    qχ[(1,1)][1] = sqrt(2) * sβ[1]
    @tensor T[1,2,3] := T[-1,-2,-3] * qχ[1,-1] * q[2,-2] * qχ[3,-3]
    @tensor C[1,2] := C[-1,-2] * qχ[1,-1] * qχ[2,-2]
    return (C, T)
end

function isingtensorz2(β)
    a = ZNTensor{Float64,4,2}((0:1,0:1,0:1,0:1),([1,1],[1,1],[1,1],[1,1]), (1,1,1,1))
    for i in Iterators.filter(iseven ∘ sum, Iterators.product(charges(a)...))
        a[i] = reshape([1],1,1,1,1)
    end
    b = ZNTensor{Float64,2,2}((0:1,0:1),([1,1],[1,1]),(1,-1))
    b[(0,0)] = sqrt(2) * reshape([sqrt(cosh(β))],1,1)
    b[(1,1)] = sqrt(2) * reshape([sqrt(sinh(β))],1,1)
    @tensor a[o1,o2,o3,o4] := a[c1,c2,c3,c4] * b[o1,c1] * b[o2,c2] * b[o3,c3] * b[o4,c4]
    return a
end

function isingmagtensorz2(β)
    a = ZNTensor{Float64,4,2}((0:1,0:1,0:1,0:1),([1,1],[1,1],[1,1],[1,1]), (1,1,1,1))
    for i in Iterators.filter(iseven ∘ sum, Iterators.product(0:1,0:1,0:1,0:1))
        a[i] = reshape([1],1,1,1,1)
    end
    b = ZNTensor{Float64,2,2}((0:1,0:1),([1,1],[1,1]),(1,-1))
    b[(0,0)] = sqrt(2) * reshape([sqrt(cosh(β))],1,1)
    b[(1,1)] = sqrt(2) * reshape([sqrt(sinh(β))],1,1)
    @tensor a[o1,o2,o3,o4] := a[c1,c2,c3,c4] * b[o1,c1] * b[o2,c2] * b[o3,c3] * b[o4,c4]
    return a
end

isingtensors(β) = (isingtensor(β), isingmagtensor(β))
isingtensorsz2(β) = (isingtensorz2(β), isingmagtensorz2(β))

# isingpart(β) = partitionfun(ising, β)

magnetisation(state::rotsymCTMState, a, asz) = magnetisation(state.C, state.T, a, asz)
magnetisation((C,T,a,asz)) = magnetisation(C,T,a,asz)

function magnetisation(C::DTensor{S,2}, T::DTensor{S,3}, a::DTensor{S,4}, asz::DTensor{S,4}) where S
    @tensor ctc[1,2,3] := C[1,-1] * T[-1,2,-2] * C[-2,3]
    @tensor begin
        mag[]  := ctc[-1,-2,-3] * T[-3,-4,-5] * ctc[-7,-6,-5] * T[-7,-8,-1] * asz[-2,-4,-6,-8]
        norm[] := ctc[-1,-2,-3] * T[-3,-4,-5] * ctc[-5,-6,-7] * T[-7,-8,-1] * a[-2,-4,-6,-8]
    end
    return scalar(mag)/scalar(norm)
end

function magnetisation(C::ZNTensor{S,2}, T::ZNTensor{S,3}, a::ZNTensor{S,4}, asz::ZNTensor{S,4}) where S
    return 0
end

function isingctm(β, χ, fixed::Bool = true; kwargs...)
    cinit, tinit = isingenvironment(β, χ, fixed = fixed)
    a, asz = isingtensors(β)
    (ctm(a, asz, χ; kwargs...)[2]..., a, asz)
end

function isingctmz2(β, χ; kwargs...)
    cinit, tinit = isingenvironmentz2(β, χ)
    a, asz = isingtensorsz2(β)
    (ctm(a, asz, χ; kwargs...)[2]..., a, asz)
end
