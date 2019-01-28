const βc = log(1+sqrt(2))/2

βofh(h) = βc/asinh(1) * asinh(sqrt(1/h))
hofβ(β) = sinh(β * asinh(1)/βc)^(-2)

function magofβ(β)
    if β > βc
        (1-sinh(2*β)^-4)^(1/8)
    else
        0
    end
end

#Crone Thesis tensors:
function isingtensor(β)
    a = DTensor{Float64}((2,2,2,2))
    initwithzero!(a)
    a[1,1,1,1] = a[2,2,2,2] = 1
    cβ, sβ = sqrt(cosh(β)), sqrt(sinh(β))
    Q = DTensor(1/sqrt(2) * [cβ+sβ cβ-sβ; cβ-sβ cβ+sβ])
    @tensor a[1,2,3,4] := a[-1,-2,-3,-4] * Q[-1,1] * Q[-2,2] * Q[-3,3] * Q[-4,4]
    return a
end

function isingmagtensor(β)
    a = DTensor{Float64}((2,2,2,2))
    initwithzero!(a)
    a[1,1,1,1] = 1
    a[2,2,2,2] = -1
    cβ, sβ = sqrt(cosh(β)), sqrt(sinh(β))
    Q = DTensor(1/sqrt(2) * [cβ+sβ cβ-sβ; cβ-sβ cβ+sβ])
    @tensor a[1,2,3,4] := a[-1,-2,-3,-4] * Q[-1,1] * Q[-2,2] * Q[-3,3] * Q[-4,4]
    return a
end

function isingenvironment(β, χ; fixed::Bool = true)
    C, T = zeros(Float64,χ,χ), zeros(Float64, χ, 2, χ)
    C[1,1] = T[1,1,1] =  1
    !fixed && (C[2,2] = T[2,2,2] = 1)

    cβ, sβ = sqrt(cosh(β)), sqrt(sinh(β))
    q = 1/sqrt(2) * [cβ+sβ cβ-sβ; cβ-sβ cβ+sβ]
    Q = zeros(Float64, χ, χ)
    Q[1:2,1:2] = q
    @tensor T[1,2,3] := T[-1,-2,-3] * Q[1,-1] * q[2,-2] * Q[3,-3]
    @tensor C[1,2] := C[-1,-2] * Q[1,-1] * Q[2,-2]
    return (DTensor(C), DTensor(T))
end

function isingenvironmentz2(β::Float64, χ::Int)
    C = DASTensor{Float64,2}(Z2(),ntuple(i -> Z2Charges(), 2),
            ([χ,χ],[χ,χ]), InOut(-1,-1))
    T = DASTensor{Float64,3}(Z2(),ntuple(i -> Z2Charges(),3),
            ([χ,χ],[1,1],[χ,χ]),InOut(1,-1,1))
    initwithzero!(C)
    initwithzero!(T)
    foreach(k -> C[k][1] = 1, keys(C))
    foreach(k -> T[k][1] = 1, keys(T))

    q =  DASTensor{Float64,2}(Z2(),(Z2Charges(), Z2Charges()),
            ([1,1],[1,1]), InOut(1,-1))
    qχ = DASTensor{Float64,2}(Z2(),(Z2Charges(), Z2Charges()),
            ([χ,χ],[χ,χ]), InOut(1,-1))
    cβ, sβ = sqrt(2) * [sqrt(cosh(β))], sqrt(2) * [sqrt(sinh(β))]
    k00 = DASSector(Z2Charge(0),Z2Charge(0))
    k11 = DASSector(Z2Charge(1),Z2Charge(1))
    q[k00]  = reshape(cβ, 1,1)
    q[k11]  = reshape(sβ, 1,1)
    qχ[k00] = zeros(Float64,χ,χ)
    qχ[k11] = zeros(Float64,χ,χ)
    qχ[k00][1] = cβ[1]
    qχ[k11][1] = sβ[1]
    @tensor T[1,2,3] := T[-1,-2,-3] * qχ[1,-1] * q[2,-2] * qχ[3,-3]
    @tensor C[1,2] := C[-1,-2] * qχ[1,-1] * qχ[2,-2]
    return (C, T)
end

function isingtensorz2(β)
    a = DASTensor{Float64,4}(Z2(),
        ntuple(i -> Z2Charges(), 4),
        ntuple(i -> [1,1], 4),
        InOut(1,1,1,1))
    for k in invariantsectors(charges(a), in_out(a))
        a[k] = reshape([1],1,1,1,1)
    end
    b = DASTensor{Float64,2}(Z2(),
        ntuple(i -> Z2Charges(), 2),
        ntuple(i -> [1,1], 2),
        InOut(1,-1))
    cβ, sβ = sqrt(2) * [sqrt(cosh(β))], sqrt(2) * [sqrt(sinh(β))]
    b[DASSector(Z2Charge(0),Z2Charge(0))] = reshape(cβ,1,1)
    b[DASSector(Z2Charge(1),Z2Charge(1))] = reshape(sβ,1,1)
    @tensor a[o1,o2,o3,o4] := a[c1,c2,c3,c4] * b[o1,c1] * b[o2,c2] * b[o3,c3] * b[o4,c4]
    return a
end

isingmagtensorz2(β) = isingtensorz2(β)

isingtensors(β) = (isingtensor(β), isingmagtensor(β))
isingtensorsz2(β) = (isingtensorz2(β), isingmagtensorz2(β))

magnetisation(state::rotsymCTMState, a, asz) = magnetisation(state.C, state.T, a, asz)
magnetisation((C,T,a,asz)) = magnetisation(C,T,a,asz)

function magnetisation(C::AbstractTensor{S,2}, T::AbstractTensor{S,3},
        a::AbstractTensor{S,4}, asz::AbstractTensor{S,4}) where S
    @tensor begin
        env[1,2,3,4] := C[-8, -1] * T[-1,3,-2] * C[-2,-3] * T[-3,2,-4] *
            C[-4,-5] * T[-5,1,-6] * C[-6,-7] * T[-7,4,-8]
        mag  = env[-1,-2,-3,-4] * asz[-1,-2,-3,-4]
        norm = env[-1,-2,-3,-4] * a[-1,-2,-3,-4]
    end
    return mag/norm
end

function isingctm(β, χ, fixed::Bool = true; kwargs...)
    cinit, tinit = isingenvironment(β, χ, fixed = fixed)
    a, asz = isingtensors(β)
    if get(kwargs, :log, false)
        c,t, info  = ctm(a, asz, χ, Cinit = cinit, Tinit = tinit; kwargs...)
        info = (β = β, info...)
        return (C = c, T = t, A = a, M = asz, ttype = :DTensor, info...)
    end
    c, t = ctm(a, asz, χ, Cinit = cinit, Tinit = tinit ; kwargs...)
    return (C = c, T = t, A = a, M = asz)
end

function isingctmz2(β, χ; kwargs...)
    cinit, tinit = isingenvironmentz2(β, χ)
    a, asz = isingtensorsz2(β)
    if get(kwargs, :log, false)
        c,t, info  = ctm(a, asz, χ, Cinit = cinit, Tinit = tinit; kwargs...)
        info = (β = β, info...)
        return (C = c, T = t, A = a, M = asz, ttype = :ZNTensor, info...)
    end
    c, t = ctm(a, asz, χ, Cinit = cinit, Tinit = tinit; kwargs...)
    return (C = c, T = t, A = a, M = asz)
end

function σxtensor()
    σx = DASTensor{Float64,2}(
            ZN{2}(),
            (Z2Charges(), Z2Charges()),
            ([1,1], [1,1]),
            InOut(1,-1))
    σx[DASSector(Z2Charge(1), Z2Charge(0))] = reshape([1], 1, 1)
    σx[DASSector(Z2Charge(0), Z2Charge(1))] = reshape([1], 1, 1)
    return σx
end
