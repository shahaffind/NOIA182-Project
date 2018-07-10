function [ G ] = loss_grad_x( X, C, W, b )
    % X - batch with size=(dim, batch_size)
    % C - correct result column for each X_i, size=(n_labels, batch_size)
    % W, b - loss layer weights
    %
    % returns G of size=(dim, batch_size)
    %   each G_i in G is the gradient of the loss function at X_i (G_i, X_i
    %   are the i-th columns on G and X)
    
    
    all_proba = softmax(X, W, b);
    
    G = W * (all_proba - C);
end
