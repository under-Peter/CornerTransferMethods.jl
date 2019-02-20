include("auxiliary-iterators.jl")

abstract type AbstractCTMIterable end
abstract type AbstractCTMState end
include("algorithms/nosym.jl")
include("algorithms/rotsym.jl")
include("algorithms/transpconjsymm.jl")

initializeC(A::DTensor{T}, χ) where T = DTensor(rand(T,χ, χ) .- 1)
initializeT(A::DTensor{T}, χ) where T = DTensor(rand(T,χ, size(A,1), χ) .-1)

function initializeC(A::DASTensor{T,4}, χ) where T
    C = checked_similar_from_indices(nothing, T, (1,2), A, :Y)
    s1, s2 = sizes(C)
    s1 .= s2 .= χ
    initwithrand!(C)
    return C
 end

function initializeT(A::DASTensor{S,4}, χ) where S
    T = checked_similar_from_indices(nothing, S, (1,2), (1,), (1,3,2), A, A, :N, :Y)
    s1, s2, s3 = sizes(T)
    s1 .= s3 .= χ
    initwithrand!(T)
    return T
 end

isconverged(state, tol) = !isempty(state.diffs) &&  state.diffs[end] < tol

function ctm(A::AbstractTensor{S,4}, Asz::AbstractTensor{S,4}, χ::Int;
                                    Cinit::Union{Nothing, AbstractTensor{S,2}} = nothing,
                                    Tinit::Union{Nothing, AbstractTensor{S,3}} = nothing,
                                    tol::Float64 = 1e-13,
                                    maxit::Int = 5000,
                                    period::Int = 100,
                                    verbose::Bool = true,
                                    log::Bool = true) where S
    stop(state) = isconverged(state,tol)
    foo(state)  = magnetisation(state, A, Asz)
    disp(state) = @printf("%5d \t| %.3e | %.3e \t| %.3e |\n",#"\t %.5e \n",
                            state[2].n_it[], state[1]/1e9,
                            state[2].diffs[end],
                            foo(state[2]))

    iter = ctmiterable(A, χ, Cinit, Tinit)
    istcsym(A)  && (iter = transconjctmiterable(A, χ, Cinit, Tinit))
    isrotsym(A) && (iter = rotsymctmiterable(A, χ, Cinit, Tinit))

    #initialize
    st,  = iterate(iter)
    iter = rest(iter, st)

    iter = halt(iter, stop)
    iter = take(iter, maxit)

    if verbose
        @printf("\tn \t| time (s)\t| diff\t\t\t| mag \n")
        iter = sample(iter, period)
        iter = stopwatch(iter)
        iter = tee(iter, disp)
        (_, state) = loop(iter)
    else
        state = loop(iter)
    end

    return  state #(state.C, state.T)
end
