function [ g ] = loss_grad_x( X, C, W, b )
    
%     [~, n_samples] = size(X);
    
    all_proba = softmax(X, W, b);
    
    g = W * (all_proba' - C);
%     g = sum(g, 2) / n_samples;
end
