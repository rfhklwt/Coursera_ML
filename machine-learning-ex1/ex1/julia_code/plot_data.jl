function plot_data(x, y)

    # ============================================================
    # ====================== YOUR CODE HERE ======================
    # Instructions: Plot the training data into a figure using the 
    #               "plot" commands. Set the axes labels using the
    #               "xlabel!" and "ylabel!" commands. Assume the po-
    #               pulation and revenue data have been passed in
    #               as the x and y arguments of this function.
    #
    # Hint: You can use the 'm' option with plot to change the markers
    #       type(e.g. m=:cross). More information please type 
    #       @show Plots.supported_markers(). Furthermore, you can make
    #       the markers larger by using plot(..., m=:cross, ms=10);
    
    p = scatter(x, y, ms = 5, label = "Training data")
    xlabel!("Profit in \$10,000s")
    ylabel!("Population of City in \$10,000s")

    return p
 
end
    