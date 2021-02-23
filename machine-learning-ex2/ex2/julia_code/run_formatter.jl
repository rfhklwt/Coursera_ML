using JuliaFormatter

for each in readdir(@__DIR__)
    (each == "run_formatter.jl" || !occursin(".jl", each)) && continue

    @info format_file(each)
end
        
