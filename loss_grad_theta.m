function [ g ] = loss_grad_theta( X, C, W, b )
    % X - batch with size=(dim, batch_size)
    % C - correct result column for each X_i, size=(n_labels, batch_size)
    % W, b - loss layer weights
    %
    % returns g with size=(number of weights, 1)
    %   g is the stochastic gradient of the loss function (w.r.t. weights)
    %   at points X

    
    all_proba = softmax(X, W, b);
    
    diff = all_proba' - C';
    g_W = X * (diff);
    g_b = sum(diff, 1);
    
    g = vertcat(g_b', g_W(:));
end

