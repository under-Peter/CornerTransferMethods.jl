using Test, CornerTransferMethods, TensorOperations, TensorNetworkTensors
@testset "CornerTransferMethods" begin
    @testset "tfisingmpo" begin
        β = randn()
        σz = [1 0; 0 -1]
        σx = [0 1; 1  0]
        id = [1 0; 0  1]
        @test ozofβ(β)[:,:] == exp(β * σz)
        oxx = oxxofβ(β, ComplexF64)
        @tensor res[1,2,3,4] := oxx[1,-1,3,-2] * oxx[2,-2,4,-1]
        ref = exp(2β*kron(σx,σx))
        resmat = fuselegs(res, ((1,2),(3,4)))[1]
        @test toarray(resmat) ≈ ref

        let β = 1e-5
            refe = exp(β * (kron(σz, id) + kron(id, σz) + 2kron(σx,σx)))
            ref = exp(β/2 * (kron(σz, id) + kron(id, σz))) *
                exp(2β * kron(σx,σx)) * #factor two probably edge-case
                exp(β/2 * (kron(σz, id) + kron(id, σz)))
            op = tfisingpropagator(β)
            @tensor op2[1,2,3,4] := op[1,-1,3,-2] *
                                    op[2,-2,4,-1];
            res = toarray(fuselegs(op2,((1,2),(3,4)))[1])
            @test ref ≈ refe && ref ≈ res
        end

        let β=1e-3
            refe = exp(β * (kron(σz,id,id) + kron(id,σz,id) + kron(id,id,σz) +
                            kron(σx,σx,id) + kron(σx,id,σx) + kron(id,σx,σx)))
            ref = exp(β/2 * (kron(σz,id,id) + kron(id,σz,id) + kron(id,id,σz))) *
                  exp(β *   (kron(σx,σx,id) + kron(id,σx,σx) + kron(σx,id,σx))) *
                  exp(β/2 * (kron(σz,id,id) + kron(id,σz,id) + kron(id,id,σz)))
            op = tfisingpropagator(β)
            @tensor op2[1,2,3,4,5,6] := op[1,-2,4,-1] *
                                        op[2,-3,5,-2] *
                                        op[3,-1,6,-3];
            res = toarray(fuselegs(op2,((1,2,3),(4,5,6)))[1])
            @test ref ≈ refe && ref ≈ res
        end

        @test kron(-σx,σx) + kron(id,-σz) + kron(-σz,id) ≈ toarray(fuselegs(tfisinghamiltonian(true),((1,2),(3,4)))[1])

        sth0 = tfisingctm(1e-3, 8, 0, maxit=10^4, period = 10^2, verbose=false)
        @test mag(sth0) ≈ 0
        @test energy(sth0) ≈ -1/2

        sth∞ = tfisingctm(1e-3, 8, 1e5, maxit=10^4, period = 10^2, verbose=false)
        @test mag(sth∞) ≈ 1
        @test energy(sth∞) ≈ -1
    end
end
