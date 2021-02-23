
#   sigmoid Compute sigmoid function
#   g = sigmoid(z) computes the sigmoid of z.

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the sigmoid of each value of z (z can be a matrix,
#               vector or scalar).

sigmoid(z) = @. 1 / (1 + ℯ^(-z))

# =============================================================
