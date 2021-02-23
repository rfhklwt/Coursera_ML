## Machine Learning Online Class - Exercise 2: Logistic Regression
@info "LOADING PACKAGE..."
using DelimitedFiles
using Printf
using Plots
using LaTeXStrings
using LinearAlgebra
using Optim
using Statistics

include("plot_data.jl")
include("map_feature.jl")
include("sigmoid.jl")
include("cost_function.jl")
include("plot_decision_boundary.jl")
include("predict.jl")
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the second part
#  of the exercise which covers regularization with logistic regression.
#
#  You will need to complete the following functions in this exericse:
#
#     sigmoid.jl
#     cost_function.jl
#     predict.jl
#     cost_function_reg.jl
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#


## Load Data
#  The first two columns contains the X values and the third column
#  contains the label (y).

data = readdlm("ex2data2.txt", ',', Float64)
X = data[:, 1:2]
y = data[:, 3]

theme(:ggplot2)
labels = ["y = 1" "y = 0"]
p = plot_data(X, y, labels)
# Put some labels
xlabel!("Microchip Test 1")
ylabel!("Microchip Test 2")
gui(p)

## =========== Part 1: Regularized Logistic Regression ============
#  In this part, you are given a dataset with data points that are not
#  linearly separable. However, you would still like to use logistic
#  regression to classify the data points.
#
#  To do so, you introduce more features to use -- in particular, you add
#  polynomial features to our data matrix (similar to polynomial
#  regression).
#

# Add Polynomial Features

# Note that mapFeature also adds a column of ones for us, so the intercept
# term is handled
X = map_feature(X[:, 1], X[:, 2])

# Initialize fitting parameters
m, n = size(X)
initial_θ = zeros(n)
grad_values = zeros(n)
# Set regularization parameter λ to 1
λ = 1

# Compute and display initial cost and gradient for regularized logistic
# regression
cost, gradient! = cost_function_reg(initial_θ, X, y, λ)

cost_val = cost(initial_θ)
gradient!(grad_values, initial_θ)

println("Cost at initial theta (zeros): ", cost_val)
println("Expected cost (approx): 0.693\n")
println("Gradient at initial theta (zeros) - first five values only:\n")
for i = 1:5
    println(grad_values[i])
end
println("\nExpected gradients (approx) - first five values only:\n")
println(" 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115")

println("\nProgram paused. Press enter to continue.\n")
readline()

# Compute and display cost and gradient
# with all-ones theta and lambda = 10
test_θ = ones(n)

test_cost, test_gradient! = cost_function_reg(initial_θ, X, y, 10)
cost_val = test_cost(test_θ)
test_gradient!(grad_values, test_θ)

println("\nCost at test theta (with lambda = 10): ", cost_val)
println("Expected cost (approx): 3.16\n")
println("Gradient at test theta - first five values only:\n")
for i = 1:5
    println(grad_values[i])
end
println("\nExpected gradients (approx) - first five values only:\n")
println(" 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922")

println("\nProgram paused. Press enter to continue.")
readline()

## ============= Part 2: Regularization and Accuracies =============
#  Optional Exercise:
#  In this part, you will get to try different values of lambda and
#  see how regularization affects the decision coundart
#
#  Try the following values of lambda (0, 1, 10, 100).
#
#  How does the decision boundary change when you vary lambda? How does
#  the training set accuracy vary?
#

# Initialize fitting parameters
initial_θ = zeros(n)

# Set regularization parameter lambda to 1 (you should vary this)
λ = 1

# Set Options
res = optimize(cost, gradient!, initial_θ, LBFGS())

# Optimize
θ = Optim.minimizer(res)

# Plot Boundary
p_1 = plot_decision_boundary(θ, X, y)
title!(string("lambda = ", λ))
xlabel!("Microchip Test 1")
ylabel!("Microchip Test 2")
gui(p_1)

# Compute accuracy on our training set
p = predict(θ, X)

println("Train Accuracy: ", mean(p .== y) * 100)
println("Expected accuracy (with lambda = 1): 83.1 (approx)\n")
