using BenchmarkTools
β = 0.42

@btime CTM.magnetisation(CTM.ctm(CTM.isingpart(β), 64,
    maxit = 100,verbose=false, tol = -1.)[2]);
