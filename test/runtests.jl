using Test, CornerTransferMethods, TensorOperations, TensorNetworkTensors, LinearAlgebra
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
        op = tfisingpropagator(1e-5)
        @tensor op2[1,2,3,4,5,6] := op[1,-2,4,-1] * op[2,-3,5,-2] * op[3,-1,6,-3];
        prop = toarray(fuselegs(op2,((1,2,3),(4,5,6)))[1])
        ham = -(kron(σz,id,id) + kron(id,σz,id) + kron(id,id,σz) +
                kron(σx,σx,id) + kron(σx,id,σx) + kron(id,σx,σx))
        λp, vecsp = eigen(prop)
        λh, vecsh = eigen(ham)
        @test vecsp[:,argmax(λp)] ≈ vecsh[:,argmin(λh)]

        @test kron(-σx,σx) + kron(id,-σz) + kron(-σz,id) ≈ toarray(fuselegs(tfisinghamiltonian(true),((1,2),(3,4)))[1])

        sth0 = tfisingctm(1e-3, 8, 0, maxit=10^4, period = 10^2, verbose=false)
        @test mag(sth0) ≈ 0
        @test energy(sth0) ≈ -1/2

        sth∞ = tfisingctm(1e-3, 8, 1e5, maxit=10^4, period = 10^2, verbose=false)
        @test mag(sth∞) ≈ 1
        @test energy(sth∞) ≈ -1
    end

    @testset "algorithms equality" begin
        A = CornerTransferMethods.isingtensor(CornerTransferMethods.βc)
        χ = 4
        rotsymiter = rotsymctmiterable(A, χ)
        tciter = transconjctmiterable(A, χ)
        iter = ctmiterable(A, χ)
        rotsymstate = ctm_kernel(rotsymiter, maxit=10^6, verbose=false)
        tcstate = ctm_kernel(tciter, maxit=10^6, verbose=false)
        state = ctm_kernel(iter, maxit=10^6, verbose=false)
        eoes = cornereoe.([state.Cs[1], state.Cs[2], state.Cs[3], state.Cs[4],
                           rotsymstate.C, tcstate.C])
        @test all((x ≈ y for x in eoes, y in eoes))

        cls = clength.([state.Ts[1], state.Ts[2], state.Ts[3], state.Ts[4],
                        rotsymstate.T, tcstate.Ts[1], tcstate.Ts[2]])
        @test all((x ≈ y for x in cls, y in cls))
    end

    @testset "UnitCell" begin
    end
end
