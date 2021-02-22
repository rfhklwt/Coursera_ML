## Machine Learning Online Class
@info "LOADING PACKAGE..."
using DelimitedFiles
using Plots
using LaTeXStrings
using LinearAlgebra
using Statistics

include("plot_data.jl")
include("compute_cost.jl")
include("gradient_descent.jl")
include("feature_normalize.jl")
include("normal_equation.jl")

#  Exercise 1: Linear regression with multiple variables
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear regression exercise. 
#
#  You will need to complete the following functions in this 
#  exericse:
#
#     plot_data.jl
#     gradient_descent.jl
#     compute_cost.jl
#     feature_normalize.jl
#     normal_equation.jl
#
#  For this part of the exercise, you will need to change some
#  parts of the code below for various experiments (e.g., changing
#  learning rates).
#

## Initialization

## ================ Part 1: Feature Normalization ================

println("Loading data ...")

## Load Data
data = readdlm("ex1data2.txt", ',', Float64)
X = data[:, 1:2]
y = data[:, 3]
m = length(y)

# Print out some data points
println("First 10 examples from the dataset: ")
for i = 1:10
    println("x = ", X[i, :], ", y = ", y[i])
end

println("Program paused. Press enter to continue.")
readline()

# Scale features and set them to zero mean
println("Normalizing Features ...\n")

X, mu, sigma = feature_normalize(X)

# Add intercept term to X
X = [ones(m, 1) X]


## ================ Part 2: Gradient Descent ================

# ====================== YOUR CODE HERE ======================
# Instructions: We have provided you with the following starter
#               code that runs gradient descent with a particular
#               learning rate (alpha). 
#
#               Your task is to first make sure that your functions - 
#               computeCost and gradientDescent already work with 
#               this starter code and support multiple variables.
#
#               After that, try running gradient descent with 
#               different values of alpha and see which one gives
#               you the best result.
#
#               Finally, you should complete the code at the end
#               to predict the price of a 1650 sq-ft, 3 br house.
#
# Hint: By using the "hold on" command, you can plot multiple
#       graphs on the same figure.
#
# Hint: At prediction, make sure you do the same feature normalization.
#

println("Running gradient descent ...")

# Choose some alpha value
alpha = 0.09
num_iters = 400

# Init Theta and Run Gradient Descent 
theta = zeros(3, 1)
theta, J_history = gradient_descent_multi(X, y, theta, alpha, num_iters)

# Plot the convergence graph
p_J = plot(1:length(J_history), J_history, legend = false)
xlabel!("Number of iterations")
ylabel!("Cost J")

gui(p_J)

# Display gradient descent"s result
println("Theta computed from gradient descent: \n")
for i = 1:length(theta)
    println(theta[i])
end

# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================
# Recall that the first column of X is all-ones. Thus, it does
# not need to be normalized.
price = dot([1 (([1650 3] .- mu) ./ sigma)], theta) # You should change this


# ============================================================

println("\nPredicted price of a 1650 sq-ft, 3 br house (using gradient descent): \n")
for i = 1:length(price)
    println(price[i])
end

println("\nProgram paused. Press enter to continue.")
readline()

## ================ Part 3: Normal Equations ================

println("Solving with normal equations...")

# ====================== YOUR CODE HERE ======================
# Instructions: The following code computes the closed form 
#               solution for linear regression using the normal
#               equations. You should complete the code in 
#               normalEqn.m
#
#               After doing so, you should complete this code 
#               to predict the price of a 1650 sq-ft, 3 br house.
#

## Load Data
data = readdlm("ex1data2.txt", ',', Float64)
X = data[:, 1:2]
y = data[:, 3]
m = length(y)

# Add intercept term to X
X = [ones(m, 1) X]

# Calculate the parameters from the normal equation
theta = normal_equation(X, y)

# Display normal equation"s result
println("Theta computed from the normal equations: \n")
for i = 1:length(theta)
    println(theta[i])
end
print("\n")

# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================
price = theta' * [1; 1650; 3] # You should change this


# ============================================================

println("Predicted price of a 1650 sq-ft, 3 br house using normal equations: \n", price)
