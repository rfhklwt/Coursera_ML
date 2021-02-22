function compute_cost(X, y, theta)
    #   compute_cost Compute cost for linear regression
    #   J = compute_cost(X, y, theta) computes the cost of using theta as the
    #   parameter for linear regression to fit the data points in X and y

    # Initialize some useful values
    m = length(y) # number of training examples

    # You need to return the following variables correctly 
    J = 0

    # ====================== YOUR CODE HERE =============================
    # Instructions: Compute the cost of a particular choice of theta
    #               You should set J to the cost.

    J = 1 / (2 * m) * dot((X * theta - y), (X * theta - y))

    # ===================================================================

end

function compute_cost_multi(X, y, theta)
    #   compute_cost_multi Compute cost for linear regression
    #   J = compute_cost_multi(X, y, theta) computes the cost of using theta as the
    #   parameter for linear regression to fit the data points in X and y

    # Initialize some useful values
    m = length(y) # number of training examples

    # You need to return the following variables correctly 
    J = 0

    # ====================== YOUR CODE HERE =============================
    # Instructions: Compute the cost of a particular choice of theta
    #               You should set J to the cost.

    J = 1 / (2 * m) * dot((X * theta - y), (X * theta - y))

    # ===================================================================

end
