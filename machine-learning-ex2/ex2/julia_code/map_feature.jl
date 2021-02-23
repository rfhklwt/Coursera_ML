function map_feature(X1, X2)
    #   map_feature Feature mapping function to polynomial features
    #
    #   map_feature(X1, X2) maps the two input features
    #   to quadratic features used in the regularization exercise.
    #
    #   Returns a new feature array with more features, comprising of 
    #   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
    #
    #   Inputs X1, X2 must be the same size
    #
    degree = 6
    out = ones(size(X1, 1), 28)
    k = 2
    for i = 1:degree, j = 0:i
        out[:, k] .= (X1 .^ (i - j)) .* (X2 .^ j)
        k += 1
    end

    out
end
