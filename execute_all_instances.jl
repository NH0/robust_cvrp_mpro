function solve_limited_time(max_time::Int, key::Int)
    start_time = time()
    print("\n--------------------------------\nstarting\n--------------------------------\n")
    r = Channel()
    @async put!(r, remotecall_fetch(solve_instances, key))
    print("Checking")
    while (time() - start_time) < max_time && !isready(r)
        sleep(40)
        print(".")
        flush(stdout)
    end
    if !isready(r)
        interrupt(key)
        println("\nInterrupted")
    else
        println("\nFinished early")
    end
end

using Distributed

if nprocs() < 2
    key = addprocs(1)[1]
else
    key = procs()[2]
end

@everywhere using CPLEX
@everywhere using JuMP
@everywhere using Dates
@everywhere include("cutting_planes.jl")

solve_limited_time(3500, key)
