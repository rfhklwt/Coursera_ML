include("plot_data.jl")
include("map_feature.jl")

function plot_decision_boundary(θ, X, y)
    #   plot_decision_boundary Plots the data points X and y into a new figure with
    #   the decision boundary defined by theta
    #   plot_decision_boundary(θ, X, y) plots the data points with + for the 
    #   positive examples and o for the negative examples. X is assumed to be 
    #   a either 
    #   1) Mx3 matrix, where the first column is an all-ones column for the 
    #      intercept.
    #   2) MxN, N>3 matrix, where the first column is all-ones

    # Plot Data
    labels = ["y = 1" "y = 0"]
    p0 = plot_data(X[:, 2:3], y, labels)

    if size(X, 2) <= 3
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = [minimum(X[:, 2]) - 2, maximum(X[:, 2]) + 2]

        # Calculate the decision boundary line
        # θ₀ + θ₁x₁ + θ₂x₂ = 0 => x₂ = -1/θ₂ * (θ₀ + θ₁x₁)
        plot_y = (-1 / θ[3]) .* (θ[1] .+ θ[2] .* plot_x)

        # Plot, and adjust axes for better viewing
        plot!(
            p0,
            plot_x,
            plot_y,
            label = "Decision Boundary",
            xlim = (30, 100),
            ylim = (30, 100),
        )
    else
        # Here is the grid range
        u = range(-1, stop = 1.5, length = 100)
        v = range(-1, stop = 1.5, length = 50)

        # Evaluate z = θ * x over the grid
        z = [dot(map_feature(ui, vi), θ) for ui in u, vi in v]

        # Plot z = 0
        # Notice you need to specify the range [0, 0]
        contour!(u, v, z', levels = [0; 0], lw = 2, label = "Decision Boundary")
    end

end
