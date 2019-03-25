"""
    ozofβ(β [, T = Float64])
return propagator exp(β σx)
"""
function ozofβ(β, T = Float64)
    σz = T[1 0; 0 -1]
    DTensor(exp(β*σz))
end

"""
    oxxofβ(β [, T = ComplexF64])
calculates the propagator for ∑_i σx_i σx_i+1, i.e.
    exp(β (∑_i σx_i σx_i+1 ))
according to Pirvu et al. 2010 _Matrix Product Operator Representations_
"""
function oxxofβ(β, T = Float64)
    id = T[1 0; 0 1]
    σx = T[0 1; 1 0]
    C0 = T[cosh(β) 0; 0 sinh(β)]
    C1 = sqrt(T(sinh(β)*cosh(β))) .* σx
    @tensor mpoxx[p1,v1,p2,v2] := C0[v1,v2] * id[p1,p2] + C1[v1,v2] * σx[p1,p2]
    return DTensor(mpoxx)
end

"""
    tfisingpropagator(β [, h = 1, T = ComplexF64])
return the MPO representing a second order trotter-suzuki expansion, i.e.
    U ≈ exp(β/2 h(σz⊗𝟙 + 𝟙⊗σz)) * exp(β σx ⊗ σx) * exp(β/2 h(σz⊗𝟙 + 𝟙⊗σz))
"""
function tfisingpropagator(β, h = 1, T = Float64)
    oz  = ozofβ(h*β/2,T)
    oxx = oxxofβ(β,T)
    @tensor op[1,2,3,4] := oz[1,-1] * oxx[-1,2,-3,4] * oz[-3,3]
    return op
end

"""
    tfisinghamiltonian([twobody = false, T = ComplexF64])
return the MPO representing the Hamiltonian
    H = -∑_i σx_i σx_i+1 - ∑_i σz_i
if `twobody=true`, return the two-body Hamiltonian
    H = -σx ⊗ σx - 1 ⊗ σz - σz ⊗ 1
"""
function tfisinghamiltonian(twobody = false, λ=1, T = Float64)
    id = T[1 0; 0 1]
    σx = T[0 1; 1 0]
    σz = T[1 0; 0 -1]

    h = DTensor(zeros(2,3,2,3))
    h[:,1,:,1] = h[:,3,:,3] =  id
    h[:,1,:,2] =  σx
    h[:,2,:,3] = -σx
    h[:,1,:,3] = -λ*σz
    twobody || return h

    rb = DTensor([0,0,1])
    lb = DTensor([1,0,0])
    @tensor op[1,2,3,4] := rb[-1] * h[1,-3,3,-1] * h[2,-2,4,-3] * lb[-2]
    return op
end

function mag(st)
    C, T = st.C, st.Ts[1]
    σz = DTensor([1 0; 0 -1])
    @tensor begin
        env[1,2] := C[-1,-2] * T[-2,2,-3] * conj(C)[-3,-4] *
                    C[-4,-5] * conj(T)[-5,1,-6] * conj(C)[-6,-1]
        m = env[-1,-2] * σz[-1,-2]
        n = env[-1,-1]
    end
    return m/n
end

function energy(st)
    C, T = st.C, st.Ts[1]
    h = tfisinghamiltonian(true)
    @tensor begin
        env[1,2,3,4] := C[-1,-2] * T[-2,3,-3] * T[-3,4,-4] * conj(C)[-4,-5] *
                        C[-5,-6] * conj(T)[-6,2,-7] * conj(T)[-7,1,-8] *
                        conj(C)[-8,-1]
        ee = env[-1,-2,-3,-4] * h[-1,-2,-3,-4]
        n  = env[-1,-2,-1,-2]
    end
    return (ee/n)/2
end

function tfisingctm(β,χ, h = 1;kwargs...)
    iter = transconjctmiterable(tfisingpropagator(β,h),χ)
    return ctm_kernel(iter; kwargs...)
end
