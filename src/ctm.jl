include("auxiliary-iterators.jl")

include("algorithms/nosym.jl")
include("algorithms/rotsym.jl")
include("algorithms/transpconjsymm.jl")

initializeC(A::DTensor{T}, χ) where T = DTensor(rand(T,χ, χ) .- 1)
initializeT(A::DTensor{T}, χ) where T = DTensor(rand(T,χ, size(A,1), χ) .-1)

function initializeC(A::DASTensor{T,4}, χ) where T
    C = checked_similar_from_indices(nothing, T, (1,2), A, :Y)
    s1, s2 = sizes(C)
    s1 .= χ
    s2 .= χ
    initwithrand!(C)
    return C
 end

function initializeT(A::DASTensor{S,4}, χ) where S
    T = checked_similar_from_indices(nothing, S, (1,2), (1,), (1,3,2), A, A, :N, :Y)
    s1, s2, s3 = sizes(T)
    s1 .= χ
    s3 .= χ
    initwithrand!(T)
    return T
 end


function ctm(A::AbstractTensor{S,4}, Asz::AbstractTensor{S,4}, χ::Int;
                                    Cinit::Union{Nothing, AbstractTensor{S,2}} = nothing,
                                    Tinit::Union{Nothing, AbstractTensor{S,3}} = nothing,
                                    tol::Float64 = 1e-13,
                                    maxit::Int = 5000,
                                    period::Int = 100,
                                    verbose::Bool = true,
                                    log::Bool = true) where S
    stop(state) = length(state.diffs) > 1 && state.diffs[end] < tol
    disp(state) = @printf("%5d \t| %.3e | %.3e \t| %.3e |\n",#"\t %.5e \n",
                            state[2][1], state[1]/1e9,
                            state[2][2].diffs[end],
                            magnetisation(state[2][2],A,Asz))

    iter = ctmiterable(A, χ, Cinit, Tinit)
    isrotsym(A) && (iter = rotsymctmiterable(A, χ, Cinit, Tinit))
    istcsym(A)  && (iter = transconjctmiterable(A, χ, Cinit, Tinit))
    iter = ctmiterable(A, χ, Cinit, Tinit)
    tol > 0 && (iter = halt(iter, stop))
    iter = take(iter, maxit)
    iter = enumerate(iter)

    if verbose
        @printf("\tn \t| time (s)\t| diff\t\t\t| mag \t\t\t| eoe\n")
        iter = sample(iter, period)
        iter = stopwatch(iter)
        iter = tee(iter, disp)
        (_, (it, state)) = loop(iter)
    else
        (it, state) = loop(iter)
    end
    # if log
    #     return (state.C, state.T, (χ = χ, n_it = it, diffs = state.diffs))
    # end
    return  state #(state.C, state.T)

end
