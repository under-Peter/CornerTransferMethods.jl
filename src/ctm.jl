include("auxiliary-iterators.jl")

abstract type AbstractCTMIterable end
abstract type AbstractCTMState end
include("algorithms/nosym.jl")
include("algorithms/rotsym.jl")
include("algorithms/transpconjsymm.jl")

initializeC(A::DTensor{T}, χ) where T = DTensor(rand(T,χ, χ) .- 1)
initializeT(A::DTensor{T}, χ, s = 1) where T = DTensor(rand(T,χ, size(A,s), χ) .-1)

function initializeC(A::DASTensor{T,4}, χ) where T
    C = checked_similar_from_indices(nothing, T, (1,2), A, :Y)
    s1, s2 = sizes(C)
    s1 .= s2 .= χ
    initwithrand!(C)
    return C
 end

function initializeT(A::DASTensor{S,4}, χ, s=1) where S
    T = checked_similar_from_indices(nothing, S, (1,2), (s,), (1,3,2), A, A, :N, :Y)
    s1, s2, s3 = sizes(T)
    s1 .= s3 .= χ
    initwithrand!(T)
    return T
 end

isconverged(state, tol) = !isempty(state.diffs) &&  state.diffs[end] < tol

function ctm(A::AbstractTensor{S,4}, Asz::AbstractTensor{S,4}, χ::Int;
                                    Cinit::Union{Nothing, AbstractTensor{S,2}} = nothing,
                                    Tinit::Union{Nothing, AbstractTensor{S,3}} = nothing,
                                    kwargs...
                                    ) where S
    if isrotsym(A)
        iter = rotsymctmiterable(A, χ, Cinit, Tinit)
    elseif istcsym(A)
        iter = transconjctmiterable(A, χ, Cinit, Tinit)
    else
        iter = ctmiterable(A, χ, Cinit, Tinit)
    end
    obs(state) = real(magnetisation(state, A, Asz))

    ctm_kernel(iter; obs = "mag" => obs, kwargs...)
end

"""
    ctm_kernel(iter::AbstractCTMIterable [, state ; tol=1e-13, maxit=10^5, period=100, verbose=true, obs=nothing)
iterates `iter` according to the options given. If a `state` is provided,
iteration starts with that state, otherwise `iter` is used as the start.
Keywords:
    - `tol`: if `tol` > 0, it is used to decide when the algorithm has converged;
       if the last element of the field `diffs` of the iterator is smaller than `tol`,
       the algorithm stops
    - `maxit`: maximum number of iterations
    - `period`: If `verbose==true`, information will be printed everye `period` iterations
    - `verbose`: if true, every `period` iterations the number of steps, the time taken since the
      start of the calculation, the last value of the `diffs` field and possibly an `obs` will be printed.
    - `obs`: a `Pair` of a string and a function on the state which will print the value of that function
      if `verbose == true`
"""
function ctm_kernel(iter::AbstractCTMIterable,
                    state::Union{Nothing, AbstractCTMState} = nothing;
                    tol::Float64 = 1e-13,
                    maxit::Int = 5000,
                    period::Int = 100,
                    verbose::Bool = true,
                    obs = nothing)
    stop(state) = isconverged(state,tol)
    function disp(state)
        ns = lpad(state[2].n_it[],7)
        ts = lpad(@sprintf("%.3E", state[1]/1e9), 13)
        ds = lpad(@sprintf("%.3E", state[2].diffs[end]),13)
        print(ns, " |", ts, " |", ds, " |")

        if !isnothing(obs)
            x = @sprintf("%.3E", obs[2](state[2]))
            print(lpad(x, 13)," |")
        end
        println()
    end

    #initialize
    isnothing(state) && (state = iterate(iter)[1])
    iter = rest(iter, state)

    iter = halt(iter, stop)
    iter = take(iter, maxit)

    if verbose
        print(lpad("n",6),"  |")
        print(lpad("t(s)", 12),"  |")
        print(lpad("diffs", 12),"  |")
        !isnothing(obs) && print(lpad(obs[1], 12),"  |")
        println()

        iter = sample(iter, period)
        iter = stopwatch(iter)
        iter = tee(iter, disp)
        (_, state) = loop(iter)
    else
        state = loop(iter)
    end

    return  state
end
