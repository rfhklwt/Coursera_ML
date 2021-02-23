function cost_function(θ, X, y)
    m = length(y)

    return (θ::Array) -> begin
        h_θ = sigmoid(X * θ)
        J = (1 / m) * sum((-y .* log.(h_θ) - (1 .- y) .* log.(1 .- h_θ)))
    end,
    (storage::Array, θ::Array) -> begin
        h_θ = sigmoid(X * θ)
        storage[:] = (1 / m) * (X' * (h_θ .- y))
    end
end

function cost_function_reg(θ, X, y, λ)
    m = length(y)

    return (θ::Array) -> begin
        h_θ = sigmoid(X * θ)
        J =
            (1 / m) * sum((-y .* log.(h_θ) - (1 .- y) .* log.(1 .- h_θ))) +
            λ / (2 * m) * sum(θ[2:end] .^ 2)
    end,
    (storage::Array, θ::Array) -> begin
        h_θ = sigmoid(X * θ)
        storage[:] = (1 / m) * (X' * (h_θ .- y)) + (λ / m) * [0; θ[2:end]]
    end
end
