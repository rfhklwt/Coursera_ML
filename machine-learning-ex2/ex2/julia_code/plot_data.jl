function plot_data(X, y, labels)
    #   plot_data Plots the data points X and y into a new figure 
    #   plot_data(x,y) plots the data points with + for the positive examples
    #   and o for the negative examples. X is assumed to be a Mx2 matrix.

    # ====================== YOUR CODE HERE ======================
    # Instructions: Plot the positive and negative examples on a
    #               2D plot, using the option ':+' for the positive
    #               examples and ':c' for the negative examples.
    #
    pos = findall(x -> x == 1, y)
    neg = findall(x -> x == 0, y)

    scatter(X[pos, 1], X[pos, 2], ms = 5, label = labels[1])
    scatter!(X[neg, 1], X[neg, 2], ms = 5, label = labels[2])

    # ============================================================

end
