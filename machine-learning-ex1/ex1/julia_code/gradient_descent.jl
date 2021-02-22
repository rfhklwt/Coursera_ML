function gradient_descent(X, y, theta, alpha, num_iters)
    #   gradient_descent Performs gradient descent to learn theta
    #   theta = gradient_descent(X, y, theta, alpha, num_iters) updates theta by 
    #   taking num_iters gradient steps with learning rate alpha

    # Initialize some useful values
    m = length(y)   # number of training examples
    J_history = zeros(num_iters)

    for iter = 1:num_iters

        # ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta. 
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCost) and gradient here.
        #

        predict = X * theta
        theta = theta - (alpha / m) * sum((predict - y) .* X, dims = 1)'

        # ============================================================

        # Save the cost J in every iteration    
        J_history[iter] = compute_cost(X, y, theta)
    end

    theta, J_history

end

function gradient_descent_multi(X, y, theta, alpha, num_iters)
    #   gradient_descent_multi Performs gradient descent to learn theta
    #   theta = gradient_descent_multi(x, y, theta, alpha, num_iters) updates theta by
    #   taking num_iters gradient steps with learning rate alpha

    # Initialize some useful values
    m = length(y)   # number of training examples
    J_history = zeros(num_iters, 1)

    for iter = 1:num_iters

        # ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta. 
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCostMulti) and gradient here.
        #

        predict = X * theta
        theta = theta - (alpha / m) * sum((predict - y) .* X, dims = 1)'

        # ============================================================

        # Save the cost J in every iteration    
        J_history[iter] = compute_cost_multi(X, y, theta)

    end

    theta, J_history

end
