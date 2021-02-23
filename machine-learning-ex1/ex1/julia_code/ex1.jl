## Machine Learning Online Class - Exercise 1: Linear Regression
@info "LOADING PACKAGE..."
using DelimitedFiles
using Plots
using LaTeXStrings
using LinearAlgebra

include("plot_data.jl")
include("compute_cost.jl")
include("gradient_descent.jl")
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exericse:
#
#     plot_data.jl
#     gradient_descent.jl
#     compute_cost.jl
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
# x refers to the population size in 10,000s
# y refers to the profit in $10,000s
#

## ======================= Part 1: Plotting =======================
println("Plotting Data ...")
data = readdlm("ex1data1.txt", ',', Float64)
X = data[:, 1]
y = data[:, 2]
m = length(y)   # number of training examples

# Plot Data
# Choice theme
theme(:ggplot2)
# Note: You have to complete the code in plot_data.jl
p = plot_data(X, y)
gui(p)

println("Program paused. Press enter to continue.")
readline()

## =================== Part 2: Cost and Gradient descent ===================

X = [ones(m) data[:, 1]]    # Add a column of ones to x
theta = zeros(2)    # initialize fitting parameters

# Some gradient descent settings
iterations = 1500
alpha = 0.01

println("\nTesting the cost function ...")
# compute and display initial cost

J = compute_cost(X, y, theta)
println("With theta = [0; 0]\nCost computed = ", J)
println("Expected cost value (approx) 32.07.")

# further testing of the cost function
J = compute_cost(X, y, [-1; 2])
println("\nWith theta = [-1 ; 2]\nCost computed = ", J)
println("Expected cost value (approx) 54.24.")

println("Program paused. Press enter to continue.")
readline();

println("\nRunning Gradient Descent ...")
# run gradient descent
theta, _ = gradient_descent(X, y, theta, alpha, iterations)

# print theta to screen
println("Theta found by gradient descent: ", theta)
println("Expected theta values (approx) [-3.6303; 1.1664]")

# Plot the linear fit
plot!(p, X[:, 2], X * theta, label = "Linear regression")   # keep previous plot visible by using plot! function
gui(p)
# Predict values for population sizes of 35,000 and 70,000
predict1 = dot([1 3.5], theta)
println("For population = 35,000, we predict a profit of ", predict1 * 10000)
predict2 = dot([1 7], theta)
println("For population = 70,000, we predict a profit of ", predict2 * 10000)

print("Program paused. Press enter to continue.\n");
readline();

## ============= Part 3: Visualizing J(theta_0, theta_1) =============
println("Visualizing J(theta_0, theta_1) ...")

# Grid over which we will calculate J
theta0_vals = range(-10, stop = 10, length = 100)
theta1_vals = range(-1, stop = 4, length = 100)

# Surface plot
p_s = surface(
    theta0_vals,
    theta1_vals,
    (theta0_vals, theta1_vals) -> compute_cost(X, y, [theta0_vals; theta1_vals]),
)
xlabel!(L"$\theta_0$")
ylabel!(L"$\theta_1$")
gui(p_s)

print("Program paused. Press enter to continue.\n")
readline()
# Contour plot
# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
p_c = contour(
    theta0_vals,
    theta1_vals,
    (theta0_vals, theta1_vals) -> compute_cost(X, y, [theta0_vals; theta1_vals]),
    levels = exp10.(range(-2, stop = 3, length = 20)),
    xlabel = L"$\theta_0$",
    ylabel = L"$\theta_1$",
)
scatter!(p_c, [theta[1]], [theta[2]], ms = 7)

gui(p_c)

println("ex1 Finished. Press ENTER to exit")
readline()
println("EXIT!")
