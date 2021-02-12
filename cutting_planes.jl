# using Distributed
using Dates

using CPLEX
using JuMP

EPSILON = 0.0001

function main_problem(static::Bool)
    sum_tij = [th[i] + th[j] for i in 1:n for j in 1:n]
    sum_tij_index = sortperm(sum_tij, rev=true)

    prod_tij = [th[i] * th[j] for i in 1:n for j in 1:n]
    prod_tij_index = sortperm(prod_tij, rev=true)

    m = Model(with_optimizer(CPLEX.Optimizer, CPXPARAM_MIP_Display=1, CPXPARAM_TimeLimit = 300))

    @variable(m, z)
    @variable(m, x[1:n,1:n], Bin)
    @variable(m, u[1:n] >=0)


    @constraint(m, z >= sum(t[i,j]*x[i,j] for i in 1:n for j in 1:n if i!=j))
    for i in 2:n
        @constraint(m, sum(x[i,j] for j in 1:n if j!=i) == 1)
        @constraint(m, sum(x[j,i] for j in 1:n if j!=i) == 1)
        @constraint(m, u[i] <= C - d[i])
        @constraint(m, u[i] <= C * (1 - x[1,i]))
        for j in 2:n
            if j != i
                @constraint(m, u[j] - u[i] >= d[i] - C*(1 - x[i,j]))
            end
        end
    end
    @constraint(m, sum(x[1,j] for j in 2:n) == sum(x[j,1] for j in 2:n))

    @objective(m, Min, z)
    
    function callback_cutting_planes(callback_data)
        x_star = zeros((n,n))
        for i in 1:n
            for j in 1:n
                x_star[i,j] = callback_value(callback_data,x[i,j])
            end
        end
        z_star = callback_value(callback_data, z)

        sub_m = sub_problem(x_star)
        optimize!(sub_m)
        z_sub = objective_value(sub_m)

        if abs(z_sub - z_star) <= EPSILON
            return
        else
            delta1_star = value.(sub_m[:delta1])
            delta2_star = value.(sub_m[:delta2])
            constraint = @build_constraint(z >= sum((t[i,j] + delta1_star[i,j]*(th[i] + th[j])
                                                     + delta2_star[i,j]*th[i]*th[j])*x[i,j]
                                                     for i in 1:n for j in 1:n))
            MOI.submit(m, MOI.LazyConstraint(callback_data), constraint)
        end
    end

    function callback_heuristic(callback_data)
        x_star = zeros((n,n))
        for i in 1:n
            for j in 1:n
                x_star[i,j] = callback_value(callback_data,x[i,j])
            end
        end
        z_star = callback_value(callback_data, z)

        delta1_array_temp = filter(index -> x_star[(index -1) ÷ n + 1, index - ((index -1) ÷ n)*n] > EPSILON, sum_tij_index)
        delta1_array = delta1_array_temp[1:min(length(delta1_array_temp), T)]

        delta2_array_temp = filter(index -> x_star[(index -1) ÷ n + 1, index - ((index -1) ÷ n)*n] > EPSILON, prod_tij_index)
        delta2_array = delta2_array_temp[1:min(length(delta2_array_temp), T*T)]
    
        z_sub = sum(sum_tij[delta1_array[i]] for i in 1:length(delta1_array))
                    + sum(2 * prod_tij[delta2_array[j]] for j in 1:length(delta2_array))
        
        if abs(z_sub - z_star) <= EPSILON
            return
        else
            constraint = @build_constraint(z >= sum(t[i,j] for i in 1:n for j in 1:n)
                                                + sum(sum_tij[delta1_array[i]] for i in 1:length(delta1_array))
                                                + sum(2 * prod_tij[delta2_array[j]] for j in 1:length(delta2_array)) )
            MOI.submit(m, MOI.LazyConstraint(callback_data), constraint)
        end
    end
    
    if (!static)
        MOI.set(m, MOI.LazyConstraintCallback(), callback_cutting_planes)
    end
    
    return m
end

function sub_problem(x_star)
    sub_m = Model(with_optimizer(CPLEX.Optimizer, CPXPARAM_MIP_Display=0))
    
    @variable(sub_m, 0 <= delta1[1:n, 1:n] <= 1)
    @variable(sub_m, 0 <= delta2[1:n, 1:n] <= 2)
    
    @constraint(sub_m, sum(delta1[i,j] for i in 1:n for j in 1:n) <= T)
    @constraint(sub_m, sum(delta2[i,j] for i in 1:n for j in 1:n) <= T*T)
    
    @objective(sub_m, Max, sum((t[i,j] + delta1[i,j]*(th[i] + th[j]) + delta2[i,j]*th[i]*th[j])*x_star[i,j]
                               for i in 1:n for j in 1:n))
    set_silent(sub_m)
    return sub_m
end

function cutting_planes()
    m = main_problem(true)
    optimize!(m)
    x_star = value.(m[:x])
    z_star = value(m[:z])
    z = m[:z]
    x = m[:x]
    sub_m = sub_problem(x_star)
    optimize!(sub_m)
    z_sub = objective_value(sub_m)

    while abs(z_sub - z_star) > EPSILON
        delta1_star = value.(sub_m[:delta1])
        delta2_star = value.(sub_m[:delta2])
        @constraint(m, z >= sum((t[i,j] + delta1_star[i,j]*(th[i] + th[j])
                                 + delta2_star[i,j]*th[i]*th[j])*x[i,j]
                                for i in 1:n for j in 1:n))
        optimize!(m)
        x_star = value.(m[:x])
        z_star = value(m[:z])
        z = m[:z]
        x = m[:x]
        sub_m = sub_problem(x_star)
        optimize!(sub_m)
        z_sub = objective_value(sub_m)
    end
    
    return round(z_star, digits = 3)
end

function branch_cut()
    m = main_problem(false)
    optimize!(m)
    z_star = JuMP.objective_value(m)
    
    return round(z_star, digits = 3)
end

function static_solve()
    m = main_problem(true)
    optimize!(m)
    z_star = JuMP.objective_value(m)
    
    return round(z_star, digits = 3)
end

function dual_solve()
    md = Model(with_optimizer(CPLEX.Optimizer))
#     set_silent(md)

    @variable(md, x[1:n, 1:n],Bin)
    @variable(md, u[1:n] >= 0)
    @variable(md, alpha >= 0)
    @variable(md, beta >= 0)
    @variable(md, gamma1[1:n, 1:n] >= 0)
    @variable(md, gamma2[1:n, 1:n] >= 0)

    @constraint(md,sum(x[1,j] for j in 2:n) == sum(x[i,1] for i in 2:n))
    for i in 2:n
        @constraint(md, sum(x[i,j] for j in 1:n if i != j) == 1)
        @constraint(md, sum(x[j,i] for j in 1:n if i != j) == 1)
        # @constraint(md,u[i] >= d[i])
        # @constraint(md,u[i] <= C)
        @constraint(md, u[i] <= C - d[i])
        @constraint(md, u[i] <= C * (1 - x[1,i]))
        for j in 2:n
            if i != j
                # @constraint(md, u[i] - u[j] + (d[j] + C) * x[i,j] <= C)
                @constraint(md, u[j] - u[i] - d[i] + C * (1 - x[i,j]) >= 0)                                
            end
        end
    end

    for i in 1:n
        for j in 1:n
            if i != j
                @constraint(md,alpha + gamma1[i,j] - (th[i] + th[j]) * x[i,j] >= 0)
                @constraint(md,beta + gamma2[i,j] - th[i] * th[j] * x[i,j] >= 0)
            end
        end
    end
    @objective(md, Min, sum(t[i,j]*x[i,j] + gamma1[i,j] + 2*gamma2[i,j]
                        for i in 1:n for j in 1:n if i != j) + T*alpha + T*T*beta)
    optimize!(md)
    z_star = JuMP.objective_value(md)
    
    return round(z_star, digits = 3)
end

function solve_instances()
    open("log"*string(Dates.now())*".txt", "w+") do io
        redirect_stdout(io) do
#             for n in 15:20
#                 filename = "n_"*string(n)*"-euclidean_true"
#                 include("data/"*filename)
# #                 time_e = @elapsed res = cutting_planes()
#                 time_e = @elapsed res = branch_cut()
# #                 time_e = @elapsed res = static_solve()
# #                 time_e = @elapsed res = dual_solve()
#                 println("\n#########\nSolved "*filename*" in "*string(time_e)*" s"*" with solution "*string(res))
#                 flush(stdout)
                
#                 filename = "n_"*string(n)*"-euclidean_false"
#                 include("data/"*filename)
# #                 time_e = @elapsed res = cutting_planes()
#                 time_e = @elapsed res = branch_cut()
# #                 time_e = @elapsed res = static_solve()
# #                 time_e = @elapsed res = dual_solve()
#                 println("\n#########\nSolved false "*filename*" in "*string(time_e)*" s"*" with solution "*string(res))
#                 flush(stdout)
#             end
            for n in range(85, step=5, stop=100)
                filename = "n_"*string(n)*"-euclidean_true"
                include("data/"*filename)
#                 time_e = @elapsed res = cutting_planes()
                time_e = @elapsed res = branch_cut()
#                 time_e = @elapsed res = static_solve()
#                 time_e = @elapsed res = dual_solve()
                println("\n#########\nSolved "*filename*" in "*string(time_e)*" s"*" with solution "*string(res))
                flush(stdout)
                
                filename = "n_"*string(n)*"-euclidean_false"
                include("data/"*filename)
#                 time_e = @elapsed res = cutting_planes()
                time_e = @elapsed res = branch_cut()
#                 time_e = @elapsed res = static_solve()
#                 time_e = @elapsed res = dual_solve()
                println("\n#########\nSolved "*filename*" in "*string(time_e)*" s"*" with solution "*string(res))
                flush(stdout)
            end
        end
    end
    return 0
end

function main()
    filename = "n_5-euclidean_true"
    include("data/"*filename)
    for i in 1:10
        # time_e = @elapsed res = cutting_planes()
        time_e = @elapsed res = branch_cut()
        println("\n#########\nSolved "*filename*" in "*string(time_e)*" s"*" with solution "*string(res))
        flush(stdout)
    end
end

solve_instances()