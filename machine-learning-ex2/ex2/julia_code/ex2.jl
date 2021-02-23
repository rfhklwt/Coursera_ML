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
include("sigmoid.jl")
include("cost_function.jl")
include("plot_decision_boundary.jl")
include("predict.jl")
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the logistic
#  regression exercise. You will need to complete the following functions 
#  in this exericse:
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
#  The first two columns contains the exam scores and the third column
#  contains the label.

data = readdlm("ex2data1.txt", ',', Float64)
X = data[:, 1:2]
y = data[:, 3]

## ==================== Part 1: Plotting ====================
#  We start the exercise by first plotting the data to understand the 
#  the problem we are working with.

println(
    "Plotting data with blue o indicating (y = 1) examples and red o indicating (y = 0) examples.",
)

theme(:ggplot2)
labels = ["Admitted" "Not admitted"]
p = plot_data(X, y, labels)
# Put some labels
xlabel!("Exam 1 score")
ylabel!("Exam 2 score")
gui(p)

println("\nProgram paused. Press enter to continue.")
readline()

## ============ Part 2: Compute Cost and Gradient ============
#  In this part of the exercise, you will implement the cost and gradient
#  for logistic regression. You neeed to complete the code in 
#  costFunction.jl

#  Setup the data matrix appropriately, and add ones for the intercept term
m, n = size(X)

# Add intercept term to x and X_test
X = [ones(m) X]

# Initialize fitting parameters
initial_θ = zeros(n + 1)
grad_values = zeros(n + 1)
# Compute and display initial cost and gradient
cost, gradient! = cost_function(initial_θ, X, y)

# Compute the cost value and gradient values
cost_val = cost(initial_θ)
gradient!(grad_values, initial_θ)

println("Cost at initial theta (zeros): ", cost_val)
println("Expected cost (approx): 0.693")
println("Gradient at initial theta (zeros): \n")
for i = 1:(n+1)
    println(grad_values[i])
end
println("\nExpected gradients (approx):\n -0.1000\n -12.0092\n -11.2628")

# Compute and display cost and gradient with non-zero theta
test_θ = [-24; 0.2; 0.2]

cost_val = cost(test_θ)
gradient!(grad_values, test_θ)

println("Cost at test theta: ", cost_val)
println("Expected cost (approx): 0.218")
println("Gradient at test theta: \n")
for i = 1:(n+1)
    println(grad_values[i])
end
println("\nExpected gradients (approx):\n 0.043\n 2.566\n 2.647")

println("\nProgram paused. Press enter to continue.")
readline()


## ============= Part 3: Optimizing using Optim.jl  =============
# The usage of Optim.jl can be found in https://julianlsolvers.github.io/Optim.jl/latest/#user/gradientsandhessians/#example
#  Set options for Optim
res = optimize(cost, gradient!, initial_θ, LBFGS())
θ = Optim.minimizer(res)

# Print theta to screen
println("Cost at theta found by Optim.jl: ", cost(θ))
println("Expected cost (approx): 0.203")
println("θ: \n")
for i = 1:(n+1)
    println(θ[i])
end
println("\nExpected theta (approx):")
println(" -25.161\n 0.206\n 0.201")

# Plot Boundary
p_1 = plot_decision_boundary(θ, X, y)
gui(p_1)

println("\nProgram paused. Press enter to continue.")
readline()

## ============== Part 4: Predict and Accuracies ==============
#  After learning the parameters, you'll like to use it to predict the outcomes
#  on unseen data. In this part, you will use the logistic regression model
#  to predict the probability that a student with score 45 on exam 1 and 
#  score 85 on exam 2 will be admitted.
#
#  Furthermore, you will compute the training and test set accuracies of 
#  our model.
#
#  Your task is to complete the code in predict.jl

#  Predict probability for a student with score 45 on exam 1 
#  and score 85 on exam 2 

prob = sigmoid(dot([1 45 85], θ))
println(
    "For a student with scores 45 and 85, we predict an admission probability of ",
    prob,
)
println("Expected value: 0.775 +/- 0.002\n")

# Compute accuracy on our training set
p = predict(θ, X)

println("Train Accuracy: ", mean(p .== y) * 100)
println("Expected accuracy (approx): 89.0\n")
