const βc = log(1+sqrt(2))/2

# function partitionfun(h, β)
#     tensor = Array{Float64, 4}(undef, 2,2,2,2)
#     for i=1:2, j=1:2, k=1:2, l=1:2
#         tensor[i,j,k,l] = exp(-β * h(i, j, k, l))
#     end
#     return tensor
# end
#
# function ising(i, j, k, l)
#     spin = (1, -1)
#     return sum(map((x,y) -> -spin[x]*spin[y], (i,j,k,l), (l,i,j,k)))
# end
#
# function ising(mag, i, j, k, l)
#     spin = (1, -1)
#     e = ising(i,j,k,l)
#     e += mag/2 * sum(x->spin[x], [i,j,k,l])
#     return e
# end
#
# ising(mag) = (x...) -> ising(mag, x...)
#Crone Thesis tensors:
function atens(β)
    a = zeros(Float64,2,2,2,2)
    a[1,1,1,1] = a[2,2,2,2]=1
    cβ = sqrt(cosh(β))
    sβ = sqrt(sinh(β))
    Q = 1/sqrt(2) * [cβ+sβ cβ-sβ; cβ-sβ cβ+sβ]
    @tensor a[1,2,3,4] := a[-1,-2,-3,-4] * Q[-1,1] * Q[-2,2] * Q[-3,3] * Q[-4,4]
    return a
end

function asztens(β)
    a = zeros(Float64,2,2,2,2)
    a[1,1,1,1] = 1
    a[2,2,2,2] = -1
    cβ = sqrt(cosh(β))
    sβ = sqrt(sinh(β))
    Q = 1/sqrt(2) * [cβ+sβ cβ-sβ; cβ-sβ cβ+sβ]
    @tensor a[1,2,3,4] := a[-1,-2,-3,-4] * Q[-1,1] * Q[-2,2] * Q[-3,3] * Q[-4,4]
    return a
end

# isingpart(β) = partitionfun(ising, β)

magnetisation(state::rotsymCTMState, a, asz) = magnetisation(state.C, state.T, a, asz)

function magnetisation(C::Array{S,2}, T::Array{S,3}, a::Array{S,4}, asz::Array{S,4}) where S
    @tensor ctc[1,2,3] := C[1,-1] * T[-1,2,-2] * C[-2,3]
    @tensor begin
        mag[]  := ctc[-1,-2,-3] * T[-3,-4,-5] * ctc[-5,-6,-7] * T[-7,-8,-1] * asz[-2,-4,-6,-8]
        norm[] := ctc[-1,-2,-3] * T[-3,-4,-5] * ctc[-5,-6,-7] * T[-7,-8,-1] * a[-2,-4,-6,-8]
    end
    return scalar(mag)/scalar(norm)
end
# magnetisation(state::rotsymCTMState) = magnetisation(state.C, state.T)
# magnetisation((C,T)) where S = magnetisation(C,T)
#
# function magnetisation(C::Array{S,2}, T::Array{S,3}) where S
#     sz = [1 0; 0 -1]
#     @tensor begin
#         o[o1,o2] := C[c1,c2] * C[c2,c3] * T[c5,o1,c3] *
#                     C[c5,c7] * C[c7,c8] * T[c8,o2,c1]
#         v[] := o[c1,c2] * sz[c2,c1]
#         n[] := o[c1,c1]
#     end
#     return scalar(v)/scalar(n)
# end
